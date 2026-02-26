#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=4

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
LOG=$BASE/log/experiment
PORT=55600

OVS_BASE=/root/data1/jinhyeok/seg123/dataset/3d_ovs
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

# ============================================================
# Part 1: Table full pipeline (RGB 3DGS + LangSplat + Render + Eval)
# ============================================================
echo "=========================================="
echo "Part 1: Table scene (full pipeline)"
echo "=========================================="

SCENE=table
SCENE_PATH=$OVS_BASE/$SCENE
MODEL_BASE=$OUT/3d_ovs/$SCENE
SCENE_LOG=$LOG/3d_ovs
mkdir -p $SCENE_LOG

# Step 1: RGB 3DGS
echo "[$SCENE] RGB 3DGS Training"
START_RGB=$SECONDS
$PYTHON $BASE/train.py -s $SCENE_PATH -m ${MODEL_BASE} --port $PORT \
    --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
    2>&1 | tee $SCENE_LOG/rgb_${SCENE}.log
RECONSTRUCTION_TIME=$(calc "$SECONDS - $START_RGB")

# Step 2: LangSplat x3
echo "[$SCENE] LangSplat Training (levels 1,2,3)"
LIFTING_TOTAL=0
for level in 1 2 3; do
    echo "[$SCENE] LangSplat level $level"
    START_LEVEL=$SECONDS
    $PYTHON $BASE/train.py -s $SCENE_PATH -m ${MODEL_BASE} \
        --start_checkpoint ${MODEL_BASE}_-1/chkpnt30000.pth \
        --feature_level $level --include_feature --port $PORT \
        --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
        2>&1 | tee $SCENE_LOG/langsplat_${SCENE}_level${level}.log
    LEVEL_TIME=$(calc "$SECONDS - $START_LEVEL")
    LIFTING_TOTAL=$(calc "$LIFTING_TOTAL + $LEVEL_TIME")
done

# Update timing
LIFTING_3D=$LIFTING_TOTAL
AE_TRAIN_TIME=108.87  # from previous run
TOTAL=$(calc "$RECONSTRUCTION_TIME + $LIFTING_3D + $AE_TRAIN_TIME")
sed -i "s/^3D-OVS|table|.*/3D-OVS|table|0|${RECONSTRUCTION_TIME}|${LIFTING_3D}|0|${AE_TRAIN_TIME}|${TOTAL}/" $RESULTS_FILE

# Step 3: Render
echo "[$SCENE] Rendering"
for level in 1 2 3; do
    $PYTHON $BASE/render.py -s $SCENE_PATH -m ${MODEL_BASE}_${level} \
        --include_feature --skip_test \
        2>&1 | tee $SCENE_LOG/render_${SCENE}_level${level}.log
done

# Step 4: Eval
echo "[$SCENE] Evaluation"
mkdir -p $CKPTS/${SCENE}/ae_ckpt
ln -sfn $BASE/autoencoder/ckpt/${SCENE}_exp/best_ckpt.pth $CKPTS/${SCENE}/ae_ckpt/best_ckpt.pth
cd $BASE/eval
$PYTHON evaluate_3dovs.py --dataset_name $SCENE --dataset_path $SCENE_PATH \
    --feat_dir $OUT/3d_ovs --ae_ckpt_dir $CKPTS --output_dir $EVAL_OUT/3d_ovs --mask_thresh 0.4 \
    2>&1 | tee $SCENE_LOG/eval_${SCENE}.log
cd $BASE

MIOU=$(grep -oP "mIoU: \K[0-9.]+" $SCENE_LOG/eval_${SCENE}.log | tail -1)
sed -i "s/^3D-OVS|table|.*/3D-OVS|table|${MIOU:-N\/A}/" $EVAL_RESULTS_FILE
echo "[$SCENE] mIoU=${MIOU:-N/A} DONE"

echo ""

# ============================================================
# Part 2: DL3DV evaluation (4 scenes)
# ============================================================
echo "=========================================="
echo "Part 2: DL3DV Evaluation"
echo "=========================================="

for SCENE in furniture_shop office park_bench_car road_car_building; do
    echo "=== $SCENE ==="
    SCENE_LOG=$LOG/dl3dv
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
    echo "[$SCENE] mIoU=${MIOU:-N/A} mACC=${MACC:-N/A}"
    echo ""
done

echo ""
echo "===== All done! ====="
echo ""
echo "=== Final Timing ==="
cat $RESULTS_FILE
echo ""
echo "=== Final Eval ==="
cat $EVAL_RESULTS_FILE
