#!/bin/bash
set -e

# ============================================================
# LangSplat Timing Measurement Script
# Runs full pipeline sequentially on GPU 4 for all scenes.
# Parses TIMING_RESULT lines and writes to Experiment.md.
# ============================================================

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
LOG=$BASE/log/timing
GPU=4
PORT=55590

mkdir -p $LOG

# Datasets and scenes
LERF_BASE=$BASE/dataset/lerf_ovs
LERF_SCENES="figurines ramen teatime waldo_kitchen"

OVS_BASE=$BASE/dataset/3d_ovs
OVS_SCENES="bed bench blue_sofa covered_desk lawn office_desk room snacks sofa table"

# Output file for collected timing results
RESULTS_FILE=$LOG/timing_results.txt
> $RESULTS_FILE

# ============================================================
# Helper: extract TIMING_RESULT values from a log file
# Usage: extract_timing <logfile> <key>
# ============================================================
extract_timing() {
    local logfile=$1
    local key=$2
    grep "TIMING_RESULT:" "$logfile" | grep -oP "${key}=\K[0-9.]+" | tail -1
}

# ============================================================
# Run pipeline for a single scene
# Args: $1=dataset_base $2=scene_name $3=model_base_name $4=dataset_label
# ============================================================
run_scene() {
    local DBASE=$1
    local SCENE=$2
    local MODEL_BASE=$3  # "gs" for LeRF, "output" for 3D-OVS
    local DATASET_LABEL=$4
    local SCENE_PATH=$DBASE/$SCENE

    echo ""
    echo "=========================================="
    echo "[$DATASET_LABEL] Processing scene: $SCENE"
    echo "=========================================="

    # --- Step 1: Preprocess (SAM + CLIP) ---
    echo "[${SCENE}] Step 1: Preprocess (SAM + CLIP)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $BASE/preprocess.py \
        --dataset_path $SCENE_PATH \
        2>&1 | tee $LOG/preprocess_${SCENE}.log

    SAM_TIME=$(extract_timing $LOG/preprocess_${SCENE}.log "SAM_TIME")
    CLIP_TIME=$(extract_timing $LOG/preprocess_${SCENE}.log "CLIP_TIME")

    # --- Step 2: RGB 3DGS Training ---
    echo "[${SCENE}] Step 2: RGB 3DGS Training (30K iterations)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $BASE/train.py \
        -s $SCENE_PATH \
        -m $SCENE_PATH/${MODEL_BASE} \
        --port $PORT \
        --checkpoint_iterations 30000 \
        --test_iterations 30000 \
        --save_iterations 30000 \
        2>&1 | tee $LOG/rgb_${SCENE}.log

    RECONSTRUCTION_TIME=$(extract_timing $LOG/rgb_${SCENE}.log "RECONSTRUCTION_TIME")

    # --- Step 3: Autoencoder Train ---
    echo "[${SCENE}] Step 3: Autoencoder Training"
    cd $BASE/autoencoder
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
        --dataset_path $SCENE_PATH \
        --dataset_name ${SCENE}_timing \
        2>&1 | tee $LOG/ae_train_${SCENE}.log
    cd $BASE

    AE_TRAIN_TIME=$(extract_timing $LOG/ae_train_${SCENE}.log "AE_TRAIN_TIME")

    # --- Step 4: Autoencoder Test (encode to dim3) ---
    echo "[${SCENE}] Step 4: Autoencoder Test (encode to dim3)"
    cd $BASE/autoencoder
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON test.py \
        --dataset_path $SCENE_PATH \
        --dataset_name ${SCENE}_timing \
        2>&1 | tee $LOG/ae_test_${SCENE}.log
    cd $BASE

    AE_TEST_TIME=$(extract_timing $LOG/ae_test_${SCENE}.log "AE_TEST_TIME")

    # --- Step 5: LangSplat Training (3 levels) ---
    echo "[${SCENE}] Step 5: LangSplat Training (levels 1,2,3)"
    LIFTING_TOTAL=0
    for level in 1 2 3; do
        echo "[${SCENE}] LangSplat level $level"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON $BASE/train.py \
            -s $SCENE_PATH \
            -m $SCENE_PATH/${MODEL_BASE} \
            --start_checkpoint $SCENE_PATH/${MODEL_BASE}_-1/chkpnt30000.pth \
            --feature_level $level \
            --include_feature \
            --port $PORT \
            --checkpoint_iterations 30000 \
            --test_iterations 30000 \
            --save_iterations 30000 \
            2>&1 | tee $LOG/langsplat_${SCENE}_level${level}.log

        LEVEL_TIME=$(extract_timing $LOG/langsplat_${SCENE}_level${level}.log "LIFTING_TRAIN_TIME")
        LIFTING_TOTAL=$(echo "$LIFTING_TOTAL + ${LEVEL_TIME:-0}" | bc)
    done

    # Compute 3D Lifting = AE_TEST_TIME + LIFTING_TOTAL
    LIFTING_3D=$(echo "${AE_TEST_TIME:-0} + $LIFTING_TOTAL" | bc)

    # Compute total
    TOTAL=$(echo "${SAM_TIME:-0} + ${RECONSTRUCTION_TIME:-0} + ${LIFTING_3D} + ${CLIP_TIME:-0} + ${AE_TRAIN_TIME:-0}" | bc)

    # Write results
    echo "${DATASET_LABEL}|${SCENE}|${SAM_TIME:-0}|${RECONSTRUCTION_TIME:-0}|${LIFTING_3D}|${CLIP_TIME:-0}|${AE_TRAIN_TIME:-0}|${TOTAL}" >> $RESULTS_FILE

    echo ""
    echo "[${SCENE}] DONE: SAM=${SAM_TIME}s RECON=${RECONSTRUCTION_TIME}s LIFTING=${LIFTING_3D}s CLIP=${CLIP_TIME}s AE_TRAIN=${AE_TRAIN_TIME}s TOTAL=${TOTAL}s"
    echo ""
}

# ============================================================
# Run all scenes
# ============================================================

echo "===== Starting Timing Measurements on GPU $GPU ====="
echo "Start time: $(date)"

# LeRF scenes
for scene in $LERF_SCENES; do
    run_scene $LERF_BASE $scene "gs" "LeRF"
done

# 3D-OVS scenes
for scene in $OVS_SCENES; do
    run_scene $OVS_BASE $scene "output" "3D-OVS"
done

echo ""
echo "End time: $(date)"

# ============================================================
# Generate Experiment.md
# ============================================================
echo "===== Generating Experiment.md ====="

fmt_time() {
    # Convert seconds to human-readable (e.g., 1234.56 -> "20m 34s")
    local secs=$1
    if [ -z "$secs" ] || [ "$secs" = "0" ]; then
        echo "N/A"
        return
    fi
    local mins=$(echo "$secs / 60" | bc)
    local remainder=$(echo "$secs - $mins * 60" | bc | xargs printf "%.0f")
    if [ "$mins" -gt 0 ]; then
        echo "${mins}m ${remainder}s"
    else
        echo "${secs}s"
    fi
}

OUTPUT=$BASE/Experiment.md

cat > $OUTPUT << 'HEADER'
# LangSplat Timing & Evaluation Results

## Timing (GPU: RTX 6000 Ada)

HEADER

# Process each dataset
for DATASET_LABEL in "LeRF" "3D-OVS"; do
    echo "### ${DATASET_LABEL} Dataset" >> $OUTPUT
    echo "" >> $OUTPUT
    echo "| Scene | SAM Extraction | 3D Reconstruction | 3D Lifting | Etc (CLIP) | Etc (AE Train) | Total |" >> $OUTPUT
    echo "|---|---|---|---|---|---|---|" >> $OUTPUT

    SUM_SAM=0; SUM_RECON=0; SUM_LIFT=0; SUM_CLIP=0; SUM_AE=0; SUM_TOTAL=0; COUNT=0

    while IFS='|' read -r label scene sam recon lift clip ae total; do
        if [ "$label" = "$DATASET_LABEL" ]; then
            echo "| ${scene} | $(fmt_time $sam) | $(fmt_time $recon) | $(fmt_time $lift) | $(fmt_time $clip) | $(fmt_time $ae) | $(fmt_time $total) |" >> $OUTPUT
            SUM_SAM=$(echo "$SUM_SAM + ${sam:-0}" | bc)
            SUM_RECON=$(echo "$SUM_RECON + ${recon:-0}" | bc)
            SUM_LIFT=$(echo "$SUM_LIFT + ${lift:-0}" | bc)
            SUM_CLIP=$(echo "$SUM_CLIP + ${clip:-0}" | bc)
            SUM_AE=$(echo "$SUM_AE + ${ae:-0}" | bc)
            SUM_TOTAL=$(echo "$SUM_TOTAL + ${total:-0}" | bc)
            COUNT=$((COUNT + 1))
        fi
    done < $RESULTS_FILE

    if [ "$COUNT" -gt 0 ]; then
        AVG_SAM=$(echo "scale=2; $SUM_SAM / $COUNT" | bc)
        AVG_RECON=$(echo "scale=2; $SUM_RECON / $COUNT" | bc)
        AVG_LIFT=$(echo "scale=2; $SUM_LIFT / $COUNT" | bc)
        AVG_CLIP=$(echo "scale=2; $SUM_CLIP / $COUNT" | bc)
        AVG_AE=$(echo "scale=2; $SUM_AE / $COUNT" | bc)
        AVG_TOTAL=$(echo "scale=2; $SUM_TOTAL / $COUNT" | bc)
        echo "| **Average** | **$(fmt_time $AVG_SAM)** | **$(fmt_time $AVG_RECON)** | **$(fmt_time $AVG_LIFT)** | **$(fmt_time $AVG_CLIP)** | **$(fmt_time $AVG_AE)** | **$(fmt_time $AVG_TOTAL)** |" >> $OUTPUT
    fi

    echo "" >> $OUTPUT
done

cat >> $OUTPUT << 'FOOTER'
## Evaluation (mIoU)

### LeRF Dataset
(TBD)

### 3D-OVS Dataset
(TBD)
FOOTER

echo "===== Experiment.md generated at $OUTPUT ====="
echo "===== All Done ====="
