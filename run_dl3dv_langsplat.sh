#!/bin/bash
set -eo pipefail

export CUDA_VISIBLE_DEVICES=4

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
LOG=$BASE/log/experiment
PORT=40365

DL3DV_BASE=/root/data1/jinhyeok/seg123/dataset/dl3dv
DL3DV_LABEL=$DL3DV_BASE/label

OUT=$BASE/output_experiment
EVAL_OUT=$BASE/eval_output_experiment
CKPTS=$BASE/ckpts_experiment

RESULTS_FILE=$LOG/timing_results.txt
EVAL_RESULTS_FILE=$LOG/eval_results.txt

calc() {
    awk "BEGIN {printf \"%.2f\", $1}"
}

echo "========== DL3DV LangSplat + Render + Eval =========="

for SCENE in furniture_shop office park_bench_car road_car_building; do
    SCENE_PATH=$DL3DV_BASE/$SCENE
    # Use correct path: -m $MODEL_BASE, code adds _${level} suffix internally
    MODEL_BASE=$OUT/dl3dv/$SCENE
    SCENE_LOG=$LOG/dl3dv

    # Checkpoint is at ${MODEL_BASE}_-1_-1/chkpnt30000.pth (from the _-1 symlink -> _-1_-1)
    CKPT_PATH=${MODEL_BASE}_-1_-1/chkpnt30000.pth

    echo ""
    echo "=========================================="
    echo "[DL3DV] $SCENE - LangSplat + Render + Eval"
    echo "=========================================="

    if [ ! -f "$CKPT_PATH" ]; then
        echo "ERROR: RGB checkpoint not found at $CKPT_PATH"
        continue
    fi

    # LangSplat Training x3
    echo "[$SCENE] LangSplat Training (levels 1,2,3)"
    LIFTING_TOTAL=0
    for level in 1 2 3; do
        echo "[$SCENE] LangSplat level $level"
        START_LEVEL=$SECONDS
        $PYTHON $BASE/train.py -s $SCENE_PATH -m ${MODEL_BASE} \
            --start_checkpoint $CKPT_PATH \
            --feature_level $level --include_feature --port $PORT \
            --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
            2>&1 | tee $SCENE_LOG/langsplat_${SCENE}_level${level}.log
        LEVEL_TIME=$(calc "$SECONDS - $START_LEVEL")
        LIFTING_TOTAL=$(calc "$LIFTING_TOTAL + $LEVEL_TIME")
    done

    # Update timing - replace LIFTING_3D in timing results
    # Read existing timing line and update LIFTING_3D field
    OLD_LINE=$(grep "^DL3DV|${SCENE}|" $RESULTS_FILE)
    if [ -n "$OLD_LINE" ]; then
        RECON_TIME=$(echo "$OLD_LINE" | cut -d'|' -f4)
        AE_TIME=$(echo "$OLD_LINE" | cut -d'|' -f7)
        TOTAL=$(calc "$RECON_TIME + $LIFTING_TOTAL + $AE_TIME")
        sed -i "s#^DL3DV|${SCENE}|.*#DL3DV|${SCENE}|0|${RECON_TIME}|${LIFTING_TOTAL}|0|${AE_TIME}|${TOTAL}#" $RESULTS_FILE
    fi

    # Rendering
    echo "[$SCENE] Rendering"
    for level in 1 2 3; do
        $PYTHON $BASE/render.py -s $SCENE_PATH -m ${MODEL_BASE}_${level} \
            --include_feature --skip_test \
            2>&1 | tee $SCENE_LOG/render_${SCENE}_level${level}.log
    done

    # Evaluation
    echo "[$SCENE] Evaluation"
    mkdir -p $CKPTS/${SCENE}/ae_ckpt
    ln -sfn $BASE/autoencoder/ckpt/${SCENE}_exp/best_ckpt.pth $CKPTS/${SCENE}/ae_ckpt/best_ckpt.pth
    cd $BASE/eval
    $PYTHON evaluate_iou_loc.py \
        --dataset_name $SCENE \
        --feat_dir $OUT/dl3dv \
        --ae_ckpt_dir $CKPTS \
        --output_dir $EVAL_OUT/dl3dv \
        --json_folder $DL3DV_LABEL \
        --mask_thresh 0.4 \
        2>&1 | tee $SCENE_LOG/eval_${SCENE}.log
    cd $BASE

    MIOU=$(grep -oP "iou chosen: \K[0-9.]+" $SCENE_LOG/eval_${SCENE}.log | tail -1)
    MACC=$(grep -oP "Localization accuracy: \K[0-9.]+" $SCENE_LOG/eval_${SCENE}.log | tail -1)
    sed -i "s/^DL3DV|${SCENE}|.*/DL3DV|${SCENE}|${MIOU:-N\/A}|${MACC:-N\/A}/" $EVAL_RESULTS_FILE
    echo "[$SCENE] mIoU=${MIOU:-N/A} mACC=${MACC:-N/A} DONE"
done

echo ""
echo "===== DL3DV Complete ====="
echo "=== Timing ===" && cat $RESULTS_FILE
echo "=== Eval ===" && cat $EVAL_RESULTS_FILE
