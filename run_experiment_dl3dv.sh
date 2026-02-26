#!/bin/bash
set -e

# ============================================================
# DL3DV experiment (no preprocess - language_features already exist)
# Sequential execution on GPU 4
# ============================================================

export CUDA_VISIBLE_DEVICES=4

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
LOG=$BASE/log/experiment
PORT=55600

DL3DV_BASE=/root/data1/jinhyeok/seg123/dataset/dl3dv
DL3DV_LABEL=$DL3DV_BASE/label

DL3DV_SCENES="furniture_shop office park_bench_car road_car_building"

OUT=$BASE/output_experiment
EVAL_OUT=$BASE/eval_output_experiment
CKPTS=$BASE/ckpts_experiment

RESULTS_FILE=$LOG/timing_results.txt
EVAL_RESULTS_FILE=$LOG/eval_results.txt

mkdir -p $LOG/dl3dv $OUT $EVAL_OUT $CKPTS

# ============================================================
# Helper functions
# ============================================================
calc() {
    awk "BEGIN {printf \"%.2f\", $1}"
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
    if [ -n "$miou" ]; then
        echo "${miou}|${macc:-N/A}"
    else
        echo "N/A"
    fi
}

# ============================================================
# DL3DV pipeline (no preprocess)
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

    local START_TOTAL=$SECONDS

    echo "[${SCENE}] Step 1: Autoencoder Training"
    local START_AE=$SECONDS
    cd $BASE/autoencoder
    $PYTHON train.py --dataset_path $SCENE_PATH --dataset_name ${SCENE}_exp \
        2>&1 | tee $SCENE_LOG/ae_train_${SCENE}.log
    cd $BASE
    local AE_TRAIN_TIME=$(calc "$SECONDS - $START_AE")

    echo "[${SCENE}] Step 2: Autoencoder Test"
    local START_AE_TEST=$SECONDS
    cd $BASE/autoencoder
    $PYTHON test.py --dataset_path $SCENE_PATH --dataset_name ${SCENE}_exp \
        2>&1 | tee $SCENE_LOG/ae_test_${SCENE}.log
    cd $BASE
    local AE_TEST_TIME=$(calc "$SECONDS - $START_AE_TEST")

    echo "[${SCENE}] Step 3: RGB 3DGS Training"
    local START_RGB=$SECONDS
    $PYTHON $BASE/train.py -s $SCENE_PATH -m ${MODEL_BASE}_-1 --port $PORT \
        --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
        2>&1 | tee $SCENE_LOG/rgb_${SCENE}.log
    local RECONSTRUCTION_TIME=$(calc "$SECONDS - $START_RGB")

    echo "[${SCENE}] Step 4: LangSplat Training (levels 1,2,3)"
    local LIFTING_TOTAL=0
    for level in 1 2 3; do
        echo "[${SCENE}] LangSplat level $level"
        local START_LEVEL=$SECONDS
        $PYTHON $BASE/train.py -s $SCENE_PATH -m ${MODEL_BASE}_${level} \
            --start_checkpoint ${MODEL_BASE}_-1/chkpnt30000.pth \
            --feature_level $level --include_feature --port $PORT \
            --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
            2>&1 | tee $SCENE_LOG/langsplat_${SCENE}_level${level}.log
        local LEVEL_TIME=$(calc "$SECONDS - $START_LEVEL")
        LIFTING_TOTAL=$(calc "$LIFTING_TOTAL + $LEVEL_TIME")
    done

    local LIFTING_3D=$(calc "$AE_TEST_TIME + $LIFTING_TOTAL")
    local TOTAL=$(calc "$SECONDS - $START_TOTAL")
    echo "DL3DV|${SCENE}|0|${RECONSTRUCTION_TIME}|${LIFTING_3D}|0|${AE_TRAIN_TIME}|${TOTAL}" >> $RESULTS_FILE

    echo "[${SCENE}] Step 5: Rendering"
    for level in 1 2 3; do
        $PYTHON $BASE/render.py -s $SCENE_PATH -m ${MODEL_BASE}_${level} \
            --include_feature --skip_test \
            2>&1 | tee $SCENE_LOG/render_${SCENE}_level${level}.log
    done

    echo "[${SCENE}] Step 6: Evaluation"
    mkdir -p $CKPTS/${SCENE}/ae_ckpt
    ln -sfn $BASE/autoencoder/ckpt/${SCENE}_exp/best_ckpt.pth $CKPTS/${SCENE}/ae_ckpt/best_ckpt.pth
    cd $BASE/eval
    $PYTHON evaluate_iou_loc.py --dataset_name $SCENE --feat_dir $OUT/dl3dv \
        --ae_ckpt_dir $CKPTS --output_dir $EVAL_OUT/dl3dv --json_folder $DL3DV_LABEL --mask_thresh 0.4 \
        2>&1 | tee $SCENE_LOG/eval_${SCENE}.log
    cd $BASE
    local EVAL_LOG=$(ls -t $EVAL_OUT/dl3dv/$SCENE/*.log 2>/dev/null | head -1)
    local EVAL_RESULT=$(extract_iou_loc "${EVAL_LOG:-$SCENE_LOG/eval_${SCENE}.log}")
    echo "DL3DV|${SCENE}|${EVAL_RESULT}" >> $EVAL_RESULTS_FILE
    echo "[${SCENE}] DONE"
}

# ============================================================
# Run all DL3DV scenes sequentially
# ============================================================
echo "========== DL3DV Dataset (4 scenes, no preprocess) =========="
for scene in $DL3DV_SCENES; do
    run_scene_dl3dv $scene
done

echo ""
echo "=========================================="
echo "DL3DV experiment complete!"
echo "=========================================="
echo "Timing results:"
grep "DL3DV" $RESULTS_FILE
echo ""
echo "Eval results:"
grep "DL3DV" $EVAL_RESULTS_FILE
