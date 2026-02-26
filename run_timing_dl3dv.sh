#!/bin/bash
set -e

# ============================================================
# LangSplat Timing for DL3DV dataset (full pipeline including preprocess)
# ============================================================

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
LOG=$BASE/log/timing_dl3dv
GPU=4
PORT=55595

mkdir -p $LOG

DL3DV_BASE=/root/data1/jinhyeok/seg123/dataset/dl3dv
DL3DV_SCENES="furniture_shop office park_bench_car road_car_building"

RESULTS_FILE=$LOG/timing_results_dl3dv.txt
> $RESULTS_FILE

extract_timing() {
    local logfile=$1
    local key=$2
    grep "TIMING_RESULT:" "$logfile" | grep -oP "${key}=\K[0-9.]+" | tail -1
}

run_scene() {
    local SCENE=$1
    local SCENE_PATH=$DL3DV_BASE/$SCENE

    echo ""
    echo "=========================================="
    echo "[DL3DV] Processing scene: $SCENE"
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
        -m $SCENE_PATH/gs_timing \
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

    # --- Step 4: Autoencoder Test ---
    echo "[${SCENE}] Step 4: Autoencoder Test"
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
            -m $SCENE_PATH/gs_timing \
            --start_checkpoint $SCENE_PATH/gs_timing_-1/chkpnt30000.pth \
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
    TOTAL=$(echo "${SAM_TIME:-0} + ${RECONSTRUCTION_TIME:-0} + ${LIFTING_3D} + ${CLIP_TIME:-0} + ${AE_TRAIN_TIME:-0}" | bc)

    echo "DL3DV|${SCENE}|${SAM_TIME:-0}|${RECONSTRUCTION_TIME:-0}|${LIFTING_3D}|${CLIP_TIME:-0}|${AE_TRAIN_TIME:-0}|${TOTAL}" >> $RESULTS_FILE

    echo ""
    echo "[${SCENE}] DONE: SAM=${SAM_TIME}s RECON=${RECONSTRUCTION_TIME}s LIFTING=${LIFTING_3D}s CLIP=${CLIP_TIME}s AE_TRAIN=${AE_TRAIN_TIME}s TOTAL=${TOTAL}s"
    echo ""
}

echo "===== Starting DL3DV Timing on GPU $GPU ====="
echo "Start time: $(date)"

for scene in $DL3DV_SCENES; do
    run_scene $scene
done

echo ""
echo "End time: $(date)"
echo "Results saved to: $RESULTS_FILE"
echo "===== All Done ====="
