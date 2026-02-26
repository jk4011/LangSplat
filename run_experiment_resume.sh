#!/bin/bash
set -e

# ============================================================
# Resume experiment from figurines level 2
# Uses same config as run_experiment.sh but skips completed work
# ============================================================

export CUDA_VISIBLE_DEVICES=4

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
LOG=$BASE/log/experiment
PORT=55600

LERF_BASE=/root/data1/jinhyeok/seg123/dataset/lerf_ovs
OVS_BASE=/root/data1/jinhyeok/seg123/dataset/3d_ovs
DL3DV_BASE=/root/data1/jinhyeok/seg123/dataset/dl3dv

LERF_LABEL=$LERF_BASE/label
DL3DV_LABEL=$DL3DV_BASE/label

LERF_SCENES="figurines ramen teatime waldo_kitchen"
OVS_SCENES="bed bench blue_sofa covered_desk lawn office_desk room snacks sofa table"
DL3DV_SCENES="furniture_shop office park_bench_car road_car_building"

OUT=$BASE/output_experiment
EVAL_OUT=$BASE/eval_output_experiment
CKPTS=$BASE/ckpts_experiment

RESULTS_FILE=$LOG/timing_results.txt
EVAL_RESULTS_FILE=$LOG/eval_results.txt

mkdir -p $LOG $OUT $EVAL_OUT $CKPTS

# Clear results files for fresh generation
> $RESULTS_FILE
> $EVAL_RESULTS_FILE

# ============================================================
# Helper functions
# ============================================================
calc() {
    awk "BEGIN {printf \"%.2f\", $1}"
}

calc4() {
    awk "BEGIN {printf \"%.4f\", $1}"
}

extract_timing() {
    local logfile=$1
    local key=$2
    grep "TIMING_RESULT:" "$logfile" | grep -oP "${key}=\K[0-9.]+" | tail -1
}

extract_iou_loc() {
    local logfile=$1
    local miou=$(grep -oP "iou chosen: \K[0-9.]+" "$logfile" | tail -1)
    local macc=$(grep -oP "Localization accuracy: \K[0-9.]+" "$logfile" | tail -1)
    echo "${miou:-N/A}|${macc:-N/A}"
}

extract_miou_3dovs() {
    local logfile=$1
    local miou=$(grep -oP "mIoU: \K[0-9.]+" "$logfile" | tail -1)
    echo "${miou:-N/A}"
}

fmt_time() {
    local secs=$1
    if [ -z "$secs" ] || [ "$secs" = "0" ] || [ "$secs" = "0.00" ] || [ "$secs" = "N/A" ]; then
        echo "N/A"
        return
    fi
    local mins=$(awk "BEGIN {printf \"%d\", $secs / 60}")
    local remainder=$(awk "BEGIN {printf \"%.0f\", $secs - $mins * 60}")
    if [ "$mins" -gt 0 ]; then
        echo "${mins}m ${remainder}s"
    else
        echo "${remainder}s"
    fi
}

# ============================================================
# Full scene pipeline (no preprocess)
# ============================================================
run_scene_full() {
    local DBASE=$1
    local SCENE=$2
    local DATASET=$3
    local DATASET_LABEL=$4
    local SCENE_PATH=$DBASE/$SCENE
    local MODEL_BASE=$OUT/$DATASET/$SCENE
    local SCENE_LOG=$LOG/$DATASET

    mkdir -p $SCENE_LOG

    echo ""
    echo "=========================================="
    echo "[$DATASET_LABEL] Processing scene: $SCENE (full)"
    echo "=========================================="

    # Step 1: AE Train
    echo "[${SCENE}] Step 1: Autoencoder Training"
    cd $BASE/autoencoder
    $PYTHON train.py --dataset_path $SCENE_PATH --dataset_name ${SCENE}_exp \
        2>&1 | tee $SCENE_LOG/ae_train_${SCENE}.log
    cd $BASE
    AE_TRAIN_TIME=$(extract_timing $SCENE_LOG/ae_train_${SCENE}.log "AE_TRAIN_TIME")

    # Step 2: AE Test
    echo "[${SCENE}] Step 2: Autoencoder Test"
    cd $BASE/autoencoder
    $PYTHON test.py --dataset_path $SCENE_PATH --dataset_name ${SCENE}_exp \
        2>&1 | tee $SCENE_LOG/ae_test_${SCENE}.log
    cd $BASE
    AE_TEST_TIME=$(extract_timing $SCENE_LOG/ae_test_${SCENE}.log "AE_TEST_TIME")

    # Step 3: RGB 3DGS
    echo "[${SCENE}] Step 3: RGB 3DGS Training"
    $PYTHON $BASE/train.py -s $SCENE_PATH -m $MODEL_BASE --port $PORT \
        --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
        2>&1 | tee $SCENE_LOG/rgb_${SCENE}.log
    RECONSTRUCTION_TIME=$(extract_timing $SCENE_LOG/rgb_${SCENE}.log "RECONSTRUCTION_TIME")

    # Step 4: LangSplat x3
    echo "[${SCENE}] Step 4: LangSplat Training (levels 1,2,3)"
    LIFTING_TOTAL=0
    for level in 1 2 3; do
        echo "[${SCENE}] LangSplat level $level"
        $PYTHON $BASE/train.py -s $SCENE_PATH -m $MODEL_BASE \
            --start_checkpoint ${MODEL_BASE}_-1/chkpnt30000.pth \
            --feature_level $level --include_feature --port $PORT \
            --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
            2>&1 | tee $SCENE_LOG/langsplat_${SCENE}_level${level}.log
        LEVEL_TIME=$(extract_timing $SCENE_LOG/langsplat_${SCENE}_level${level}.log "LIFTING_TRAIN_TIME")
        LIFTING_TOTAL=$(calc "$LIFTING_TOTAL + ${LEVEL_TIME:-0}")
    done

    LIFTING_3D=$(calc "${AE_TEST_TIME:-0} + $LIFTING_TOTAL")
    TOTAL=$(calc "${RECONSTRUCTION_TIME:-0} + ${LIFTING_3D} + ${AE_TRAIN_TIME:-0}")
    echo "${DATASET_LABEL}|${SCENE}|0|${RECONSTRUCTION_TIME:-0}|${LIFTING_3D}|0|${AE_TRAIN_TIME:-0}|${TOTAL}" >> $RESULTS_FILE

    # Step 5: Render
    echo "[${SCENE}] Step 5: Rendering"
    for level in 1 2 3; do
        $PYTHON $BASE/render.py -s $SCENE_PATH -m ${MODEL_BASE}_${level} \
            --include_feature --skip_test \
            2>&1 | tee $SCENE_LOG/render_${SCENE}_level${level}.log
    done

    # Step 6: Evaluate
    echo "[${SCENE}] Step 6: Evaluation"
    mkdir -p $CKPTS/${SCENE}/ae_ckpt
    ln -sfn $BASE/autoencoder/ckpt/${SCENE}_exp/best_ckpt.pth $CKPTS/${SCENE}/ae_ckpt/best_ckpt.pth

    if [ "$DATASET" = "lerf" ]; then
        cd $BASE/eval
        $PYTHON evaluate_iou_loc.py --dataset_name $SCENE --feat_dir $OUT/$DATASET \
            --ae_ckpt_dir $CKPTS --output_dir $EVAL_OUT/$DATASET --json_folder $LERF_LABEL --mask_thresh 0.4 \
            2>&1 | tee $SCENE_LOG/eval_${SCENE}.log
        cd $BASE
        EVAL_LOG=$(ls -t $EVAL_OUT/$DATASET/$SCENE/*.log 2>/dev/null | head -1)
        EVAL_RESULT=$(extract_iou_loc "${EVAL_LOG:-$SCENE_LOG/eval_${SCENE}.log}")
        echo "${DATASET_LABEL}|${SCENE}|${EVAL_RESULT}" >> $EVAL_RESULTS_FILE
    elif [ "$DATASET" = "3d_ovs" ]; then
        cd $BASE/eval
        $PYTHON evaluate_3dovs.py --dataset_name $SCENE --dataset_path $SCENE_PATH \
            --feat_dir $OUT/$DATASET --ae_ckpt_dir $CKPTS --output_dir $EVAL_OUT/$DATASET --mask_thresh 0.4 \
            2>&1 | tee $SCENE_LOG/eval_${SCENE}.log
        cd $BASE
        EVAL_LOG=$(ls -t $EVAL_OUT/$DATASET/$SCENE/*.log 2>/dev/null | head -1)
        MIOU=$(extract_miou_3dovs "${EVAL_LOG:-$SCENE_LOG/eval_${SCENE}.log}")
        echo "${DATASET_LABEL}|${SCENE}|${MIOU}" >> $EVAL_RESULTS_FILE
    fi
    echo "[${SCENE}] DONE"
}

# ============================================================
# DL3DV pipeline (with preprocess)
# ============================================================
run_scene_dl3dv() {
    local SCENE=$1
    local SCENE_PATH=$DL3DV_BASE/$SCENE
    local MODEL_BASE=$OUT/dl3dv/$SCENE
    local SCENE_LOG=$LOG/dl3dv
    mkdir -p $SCENE_LOG

    echo ""
    echo "=========================================="
    echo "[DL3DV] Processing scene: $SCENE"
    echo "=========================================="

    echo "[${SCENE}] Step 1: Preprocess (SAM + CLIP)"
    $PYTHON $BASE/preprocess.py --dataset_path $SCENE_PATH \
        2>&1 | tee $SCENE_LOG/preprocess_${SCENE}.log
    SAM_TIME=$(extract_timing $SCENE_LOG/preprocess_${SCENE}.log "SAM_TIME")
    CLIP_TIME=$(extract_timing $SCENE_LOG/preprocess_${SCENE}.log "CLIP_TIME")

    echo "[${SCENE}] Step 2: Autoencoder Training"
    cd $BASE/autoencoder
    $PYTHON train.py --dataset_path $SCENE_PATH --dataset_name ${SCENE}_exp \
        2>&1 | tee $SCENE_LOG/ae_train_${SCENE}.log
    cd $BASE
    AE_TRAIN_TIME=$(extract_timing $SCENE_LOG/ae_train_${SCENE}.log "AE_TRAIN_TIME")

    echo "[${SCENE}] Step 3: Autoencoder Test"
    cd $BASE/autoencoder
    $PYTHON test.py --dataset_path $SCENE_PATH --dataset_name ${SCENE}_exp \
        2>&1 | tee $SCENE_LOG/ae_test_${SCENE}.log
    cd $BASE
    AE_TEST_TIME=$(extract_timing $SCENE_LOG/ae_test_${SCENE}.log "AE_TEST_TIME")

    echo "[${SCENE}] Step 4: RGB 3DGS Training"
    $PYTHON $BASE/train.py -s $SCENE_PATH -m $MODEL_BASE --port $PORT \
        --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
        2>&1 | tee $SCENE_LOG/rgb_${SCENE}.log
    RECONSTRUCTION_TIME=$(extract_timing $SCENE_LOG/rgb_${SCENE}.log "RECONSTRUCTION_TIME")

    echo "[${SCENE}] Step 5: LangSplat Training (levels 1,2,3)"
    LIFTING_TOTAL=0
    for level in 1 2 3; do
        echo "[${SCENE}] LangSplat level $level"
        $PYTHON $BASE/train.py -s $SCENE_PATH -m $MODEL_BASE \
            --start_checkpoint ${MODEL_BASE}_-1/chkpnt30000.pth \
            --feature_level $level --include_feature --port $PORT \
            --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
            2>&1 | tee $SCENE_LOG/langsplat_${SCENE}_level${level}.log
        LEVEL_TIME=$(extract_timing $SCENE_LOG/langsplat_${SCENE}_level${level}.log "LIFTING_TRAIN_TIME")
        LIFTING_TOTAL=$(calc "$LIFTING_TOTAL + ${LEVEL_TIME:-0}")
    done

    LIFTING_3D=$(calc "${AE_TEST_TIME:-0} + $LIFTING_TOTAL")
    ETC_TIME=$(calc "${CLIP_TIME:-0} + ${AE_TRAIN_TIME:-0}")
    TOTAL=$(calc "${SAM_TIME:-0} + ${RECONSTRUCTION_TIME:-0} + ${LIFTING_3D} + ${ETC_TIME}")
    echo "DL3DV|${SCENE}|${SAM_TIME:-0}|${RECONSTRUCTION_TIME:-0}|${LIFTING_3D}|${CLIP_TIME:-0}|${AE_TRAIN_TIME:-0}|${TOTAL}" >> $RESULTS_FILE

    echo "[${SCENE}] Step 6: Rendering"
    for level in 1 2 3; do
        $PYTHON $BASE/render.py -s $SCENE_PATH -m ${MODEL_BASE}_${level} \
            --include_feature --skip_test \
            2>&1 | tee $SCENE_LOG/render_${SCENE}_level${level}.log
    done

    echo "[${SCENE}] Step 7: Evaluation"
    mkdir -p $CKPTS/${SCENE}/ae_ckpt
    ln -sfn $BASE/autoencoder/ckpt/${SCENE}_exp/best_ckpt.pth $CKPTS/${SCENE}/ae_ckpt/best_ckpt.pth
    cd $BASE/eval
    $PYTHON evaluate_iou_loc.py --dataset_name $SCENE --feat_dir $OUT/dl3dv \
        --ae_ckpt_dir $CKPTS --output_dir $EVAL_OUT/dl3dv --json_folder $DL3DV_LABEL --mask_thresh 0.4 \
        2>&1 | tee $SCENE_LOG/eval_${SCENE}.log
    cd $BASE
    EVAL_LOG=$(ls -t $EVAL_OUT/dl3dv/$SCENE/*.log 2>/dev/null | head -1)
    EVAL_RESULT=$(extract_iou_loc "${EVAL_LOG:-$SCENE_LOG/eval_${SCENE}.log}")
    echo "DL3DV|${SCENE}|${EVAL_RESULT}" >> $EVAL_RESULTS_FILE
    echo "[${SCENE}] DONE"
}

# ============================================================
# RESUME: figurines from level 2 onwards
# ============================================================
echo "===== Resuming LangSplat Experiment ====="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

SCENE="figurines"
SCENE_PATH=$LERF_BASE/$SCENE
MODEL_BASE=$OUT/lerf/$SCENE
SCENE_LOG=$LOG/lerf
mkdir -p $SCENE_LOG

echo "=========================================="
echo "[LeRF] Resuming figurines: levels 2,3 + render + eval"
echo "=========================================="

# Already completed for figurines: AE train, AE test, RGB 3DGS, LangSplat level 1
AE_TRAIN_TIME=$(extract_timing $SCENE_LOG/ae_train_${SCENE}.log "AE_TRAIN_TIME")
AE_TEST_TIME=$(extract_timing $SCENE_LOG/ae_test_${SCENE}.log "AE_TEST_TIME")
RECONSTRUCTION_TIME=$(extract_timing $SCENE_LOG/rgb_${SCENE}.log "RECONSTRUCTION_TIME")
LEVEL1_TIME=$(extract_timing $SCENE_LOG/langsplat_${SCENE}_level1.log "LIFTING_TRAIN_TIME")

LIFTING_TOTAL=$LEVEL1_TIME

# LangSplat levels 2 and 3
for level in 2 3; do
    echo "[${SCENE}] LangSplat level $level"
    $PYTHON $BASE/train.py -s $SCENE_PATH -m $MODEL_BASE \
        --start_checkpoint ${MODEL_BASE}_-1/chkpnt30000.pth \
        --feature_level $level --include_feature --port $PORT \
        --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
        2>&1 | tee $SCENE_LOG/langsplat_${SCENE}_level${level}.log
    LEVEL_TIME=$(extract_timing $SCENE_LOG/langsplat_${SCENE}_level${level}.log "LIFTING_TRAIN_TIME")
    LIFTING_TOTAL=$(calc "$LIFTING_TOTAL + ${LEVEL_TIME:-0}")
done

LIFTING_3D=$(calc "${AE_TEST_TIME:-0} + $LIFTING_TOTAL")
TOTAL=$(calc "${RECONSTRUCTION_TIME:-0} + ${LIFTING_3D} + ${AE_TRAIN_TIME:-0}")
echo "LeRF|${SCENE}|0|${RECONSTRUCTION_TIME:-0}|${LIFTING_3D}|0|${AE_TRAIN_TIME:-0}|${TOTAL}" >> $RESULTS_FILE

# Render
echo "[${SCENE}] Rendering (3 levels)"
for level in 1 2 3; do
    $PYTHON $BASE/render.py -s $SCENE_PATH -m ${MODEL_BASE}_${level} \
        --include_feature --skip_test \
        2>&1 | tee $SCENE_LOG/render_${SCENE}_level${level}.log
done

# Evaluate
echo "[${SCENE}] Evaluation"
mkdir -p $CKPTS/${SCENE}/ae_ckpt
ln -sfn $BASE/autoencoder/ckpt/${SCENE}_exp/best_ckpt.pth $CKPTS/${SCENE}/ae_ckpt/best_ckpt.pth
cd $BASE/eval
$PYTHON evaluate_iou_loc.py --dataset_name $SCENE --feat_dir $OUT/lerf \
    --ae_ckpt_dir $CKPTS --output_dir $EVAL_OUT/lerf --json_folder $LERF_LABEL --mask_thresh 0.4 \
    2>&1 | tee $SCENE_LOG/eval_${SCENE}.log
cd $BASE
EVAL_LOG=$(ls -t $EVAL_OUT/lerf/$SCENE/*.log 2>/dev/null | head -1)
EVAL_RESULT=$(extract_iou_loc "${EVAL_LOG:-$SCENE_LOG/eval_${SCENE}.log}")
echo "LeRF|${SCENE}|${EVAL_RESULT}" >> $EVAL_RESULTS_FILE

echo "[figurines] DONE (resumed)"
echo ""

# ============================================================
# Continue with remaining LeRF scenes
# ============================================================
for scene in ramen teatime waldo_kitchen; do
    run_scene_full $LERF_BASE $scene "lerf" "LeRF"
done

# 3D-OVS scenes
echo "========== 3D-OVS Dataset (10 scenes) =========="
for scene in $OVS_SCENES; do
    run_scene_full $OVS_BASE $scene "3d_ovs" "3D-OVS"
done

# DL3DV scenes
echo "========== DL3DV Dataset (4 scenes) =========="
for scene in $DL3DV_SCENES; do
    run_scene_dl3dv $scene
done

echo ""
echo "End time: $(date)"

# ============================================================
# Generate Experiment.md (same as run_experiment.sh)
# ============================================================
echo "===== Generating Experiment.md ====="

OUTPUT=$BASE/Experiment.md

cat > $OUTPUT << 'HEADER'
# LangSplat Experiment Results

HEADER

cat >> $OUTPUT << 'EOF'
## Timing Results

### LeRF Dataset

| Scene | 3D Reconstruction | 3D Lifting | Etc (AE Train) | Total |
|---|---|---|---|---|
EOF

SUM_RECON=0; SUM_LIFT=0; SUM_AE=0; SUM_TOTAL=0; COUNT=0
while IFS='|' read -r label scene sam recon lift clip ae total; do
    if [ "$label" = "LeRF" ]; then
        echo "| ${scene} | $(fmt_time $recon) | $(fmt_time $lift) | $(fmt_time $ae) | $(fmt_time $total) |" >> $OUTPUT
        SUM_RECON=$(calc "$SUM_RECON + ${recon:-0}")
        SUM_LIFT=$(calc "$SUM_LIFT + ${lift:-0}")
        SUM_AE=$(calc "$SUM_AE + ${ae:-0}")
        SUM_TOTAL=$(calc "$SUM_TOTAL + ${total:-0}")
        COUNT=$((COUNT + 1))
    fi
done < $RESULTS_FILE

if [ "$COUNT" -gt 0 ]; then
    AVG_RECON=$(calc "$SUM_RECON / $COUNT")
    AVG_LIFT=$(calc "$SUM_LIFT / $COUNT")
    AVG_AE=$(calc "$SUM_AE / $COUNT")
    AVG_TOTAL=$(calc "$SUM_TOTAL / $COUNT")
    echo "| **Mean** | **$(fmt_time $AVG_RECON)** | **$(fmt_time $AVG_LIFT)** | **$(fmt_time $AVG_AE)** | **$(fmt_time $AVG_TOTAL)** |" >> $OUTPUT
fi
echo "" >> $OUTPUT

cat >> $OUTPUT << 'EOF'
### 3D-OVS Dataset

| Scene | 3D Reconstruction | 3D Lifting | Etc (AE Train) | Total |
|---|---|---|---|---|
EOF

SUM_RECON=0; SUM_LIFT=0; SUM_AE=0; SUM_TOTAL=0; COUNT=0
while IFS='|' read -r label scene sam recon lift clip ae total; do
    if [ "$label" = "3D-OVS" ]; then
        echo "| ${scene} | $(fmt_time $recon) | $(fmt_time $lift) | $(fmt_time $ae) | $(fmt_time $total) |" >> $OUTPUT
        SUM_RECON=$(calc "$SUM_RECON + ${recon:-0}")
        SUM_LIFT=$(calc "$SUM_LIFT + ${lift:-0}")
        SUM_AE=$(calc "$SUM_AE + ${ae:-0}")
        SUM_TOTAL=$(calc "$SUM_TOTAL + ${total:-0}")
        COUNT=$((COUNT + 1))
    fi
done < $RESULTS_FILE

if [ "$COUNT" -gt 0 ]; then
    AVG_RECON=$(calc "$SUM_RECON / $COUNT")
    AVG_LIFT=$(calc "$SUM_LIFT / $COUNT")
    AVG_AE=$(calc "$SUM_AE / $COUNT")
    AVG_TOTAL=$(calc "$SUM_TOTAL / $COUNT")
    echo "| **Mean** | **$(fmt_time $AVG_RECON)** | **$(fmt_time $AVG_LIFT)** | **$(fmt_time $AVG_AE)** | **$(fmt_time $AVG_TOTAL)** |" >> $OUTPUT
fi
echo "" >> $OUTPUT

cat >> $OUTPUT << 'EOF'
### DL3DV Dataset

| Scene | SAM Inference | 3D Reconstruction | 3D Lifting | Etc (CLIP + AE Train) | Total |
|---|---|---|---|---|---|
EOF

SUM_SAM=0; SUM_RECON=0; SUM_LIFT=0; SUM_ETC=0; SUM_TOTAL=0; COUNT=0
while IFS='|' read -r label scene sam recon lift clip ae total; do
    if [ "$label" = "DL3DV" ]; then
        ETC=$(calc "${clip:-0} + ${ae:-0}")
        echo "| ${scene} | $(fmt_time $sam) | $(fmt_time $recon) | $(fmt_time $lift) | $(fmt_time $ETC) | $(fmt_time $total) |" >> $OUTPUT
        SUM_SAM=$(calc "$SUM_SAM + ${sam:-0}")
        SUM_RECON=$(calc "$SUM_RECON + ${recon:-0}")
        SUM_LIFT=$(calc "$SUM_LIFT + ${lift:-0}")
        SUM_ETC=$(calc "$SUM_ETC + $ETC")
        SUM_TOTAL=$(calc "$SUM_TOTAL + ${total:-0}")
        COUNT=$((COUNT + 1))
    fi
done < $RESULTS_FILE

if [ "$COUNT" -gt 0 ]; then
    AVG_SAM=$(calc "$SUM_SAM / $COUNT")
    AVG_RECON=$(calc "$SUM_RECON / $COUNT")
    AVG_LIFT=$(calc "$SUM_LIFT / $COUNT")
    AVG_ETC=$(calc "$SUM_ETC / $COUNT")
    AVG_TOTAL=$(calc "$SUM_TOTAL / $COUNT")
    echo "| **Mean** | **$(fmt_time $AVG_SAM)** | **$(fmt_time $AVG_RECON)** | **$(fmt_time $AVG_LIFT)** | **$(fmt_time $AVG_ETC)** | **$(fmt_time $AVG_TOTAL)** |" >> $OUTPUT
fi
echo "" >> $OUTPUT

cat >> $OUTPUT << 'EOF'
## Evaluation Results

### LeRF Dataset

| Scene | mIoU | mACC (Loc.) |
|---|---|---|
EOF

SUM_IOU=0; SUM_ACC=0; COUNT=0
while IFS='|' read -r label scene miou macc; do
    if [ "$label" = "LeRF" ]; then
        echo "| ${scene} | ${miou} | ${macc} |" >> $OUTPUT
        if [ "$miou" != "N/A" ]; then
            SUM_IOU=$(calc "$SUM_IOU + $miou")
            SUM_ACC=$(calc "$SUM_ACC + $macc")
            COUNT=$((COUNT + 1))
        fi
    fi
done < $EVAL_RESULTS_FILE

if [ "$COUNT" -gt 0 ]; then
    AVG_IOU=$(calc4 "$SUM_IOU / $COUNT")
    AVG_ACC=$(calc4 "$SUM_ACC / $COUNT")
    echo "| **Mean** | **${AVG_IOU}** | **${AVG_ACC}** |" >> $OUTPUT
fi
echo "" >> $OUTPUT

cat >> $OUTPUT << 'EOF'
### 3D-OVS Dataset

| Scene | mIoU |
|---|---|
EOF

SUM_IOU=0; COUNT=0
while IFS='|' read -r label scene miou; do
    if [ "$label" = "3D-OVS" ]; then
        echo "| ${scene} | ${miou} |" >> $OUTPUT
        if [ "$miou" != "N/A" ]; then
            SUM_IOU=$(calc "$SUM_IOU + $miou")
            COUNT=$((COUNT + 1))
        fi
    fi
done < $EVAL_RESULTS_FILE

if [ "$COUNT" -gt 0 ]; then
    AVG_IOU=$(calc4 "$SUM_IOU / $COUNT")
    echo "| **Mean** | **${AVG_IOU}** |" >> $OUTPUT
fi
echo "" >> $OUTPUT

cat >> $OUTPUT << 'EOF'
### DL3DV Dataset

| Scene | mIoU | mACC (Loc.) |
|---|---|---|
EOF

SUM_IOU=0; SUM_ACC=0; COUNT=0
while IFS='|' read -r label scene miou macc; do
    if [ "$label" = "DL3DV" ]; then
        echo "| ${scene} | ${miou} | ${macc} |" >> $OUTPUT
        if [ "$miou" != "N/A" ]; then
            SUM_IOU=$(calc "$SUM_IOU + $miou")
            SUM_ACC=$(calc "$SUM_ACC + $macc")
            COUNT=$((COUNT + 1))
        fi
    fi
done < $EVAL_RESULTS_FILE

if [ "$COUNT" -gt 0 ]; then
    AVG_IOU=$(calc4 "$SUM_IOU / $COUNT")
    AVG_ACC=$(calc4 "$SUM_ACC / $COUNT")
    echo "| **Mean** | **${AVG_IOU}** | **${AVG_ACC}** |" >> $OUTPUT
fi
echo "" >> $OUTPUT

echo "===== Experiment.md generated at $OUTPUT ====="
echo "===== All Done ====="
