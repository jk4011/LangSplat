#!/bin/bash
set -e

# ============================================================
# LangSplat Timing (Skip Preprocess) - Measure remaining stages
# ============================================================

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
LOG=$BASE/log/timing
GPU=4
PORT=55590

mkdir -p $LOG

LERF_BASE=$BASE/dataset/lerf_ovs
LERF_SCENES="figurines ramen teatime waldo_kitchen"

OVS_BASE=$BASE/dataset/3d_ovs
OVS_SCENES="bed bench blue_sofa covered_desk lawn office_desk room snacks sofa table"

RESULTS_FILE=$LOG/timing_results_no_preprocess.txt
> $RESULTS_FILE

extract_timing() {
    local logfile=$1
    local key=$2
    grep "TIMING_RESULT:" "$logfile" | grep -oP "${key}=\K[0-9.]+" | tail -1
}

run_scene() {
    local DBASE=$1
    local SCENE=$2
    local MODEL_BASE=$3
    local DATASET_LABEL=$4
    local SCENE_PATH=$DBASE/$SCENE

    echo ""
    echo "=========================================="
    echo "[$DATASET_LABEL] Processing scene: $SCENE"
    echo "=========================================="

    # --- Step 1: RGB 3DGS Training ---
    echo "[${SCENE}] Step 1: RGB 3DGS Training (30K iterations)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $BASE/train.py \
        -s $SCENE_PATH \
        -m $SCENE_PATH/${MODEL_BASE}_timing \
        --port $PORT \
        --checkpoint_iterations 30000 \
        --test_iterations 30000 \
        --save_iterations 30000 \
        2>&1 | tee $LOG/rgb_${SCENE}.log

    RECONSTRUCTION_TIME=$(extract_timing $LOG/rgb_${SCENE}.log "RECONSTRUCTION_TIME")

    # --- Step 2: Autoencoder Train ---
    echo "[${SCENE}] Step 2: Autoencoder Training"
    cd $BASE/autoencoder
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
        --dataset_path $SCENE_PATH \
        --dataset_name ${SCENE}_timing \
        2>&1 | tee $LOG/ae_train_${SCENE}.log
    cd $BASE

    AE_TRAIN_TIME=$(extract_timing $LOG/ae_train_${SCENE}.log "AE_TRAIN_TIME")

    # --- Step 3: Autoencoder Test ---
    echo "[${SCENE}] Step 3: Autoencoder Test"
    cd $BASE/autoencoder
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON test.py \
        --dataset_path $SCENE_PATH \
        --dataset_name ${SCENE}_timing \
        2>&1 | tee $LOG/ae_test_${SCENE}.log
    cd $BASE

    AE_TEST_TIME=$(extract_timing $LOG/ae_test_${SCENE}.log "AE_TEST_TIME")

    # --- Step 4: LangSplat Training (3 levels) ---
    echo "[${SCENE}] Step 4: LangSplat Training (levels 1,2,3)"
    LIFTING_TOTAL=0
    for level in 1 2 3; do
        echo "[${SCENE}] LangSplat level $level"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON $BASE/train.py \
            -s $SCENE_PATH \
            -m $SCENE_PATH/${MODEL_BASE}_timing \
            --start_checkpoint $SCENE_PATH/${MODEL_BASE}_timing_-1/chkpnt30000.pth \
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

    LIFTING_3D=$(echo "${AE_TEST_TIME:-0} + $LIFTING_TOTAL" | bc)
    TOTAL=$(echo "${RECONSTRUCTION_TIME:-0} + ${LIFTING_3D} + ${AE_TRAIN_TIME:-0}" | bc)

    echo "${DATASET_LABEL}|${SCENE}|${RECONSTRUCTION_TIME:-0}|${LIFTING_3D}|${AE_TRAIN_TIME:-0}|${TOTAL}" >> $RESULTS_FILE

    echo ""
    echo "[${SCENE}] DONE: RECON=${RECONSTRUCTION_TIME}s LIFTING=${LIFTING_3D}s AE_TRAIN=${AE_TRAIN_TIME}s TOTAL=${TOTAL}s"
    echo ""
}

echo "===== Starting Timing (no preprocess) on GPU $GPU ====="
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
echo "Results saved to: $RESULTS_FILE"
echo "===== All Done ====="
