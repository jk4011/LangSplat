#!/bin/bash
set -e

# ============================================================
# Resume experiment #2: Re-run failed evals + table + DL3DV
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

OUT=$BASE/output_experiment
EVAL_OUT=$BASE/eval_output_experiment
CKPTS=$BASE/ckpts_experiment

RESULTS_FILE=$LOG/timing_results.txt
EVAL_RESULTS_FILE=$LOG/eval_results.txt

mkdir -p $LOG $OUT $EVAL_OUT $CKPTS

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
    echo "${miou:-N/A}|${macc:-N/A}"
}

extract_miou_3dovs() {
    local logfile=$1
    local miou=$(grep -oP "mIoU: \K[0-9.]+" "$logfile" | tail -1)
    echo "${miou:-N/A}"
}

echo "===== Resume Experiment #2 ====="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# ============================================================
# Part 1: Re-run failed 3D-OVS evals
# (bed, bench, blue_sofa, covered_desk, lawn)
# Already trained + rendered, just need eval
# ============================================================
echo "=========================================="
echo "Part 1: Re-running failed 3D-OVS evals"
echo "=========================================="

for SCENE in bed bench blue_sofa covered_desk lawn; do
    SCENE_PATH=$OVS_BASE/$SCENE
    SCENE_LOG=$LOG/3d_ovs

    echo "[3D-OVS] Re-evaluating $SCENE"
    mkdir -p $CKPTS/${SCENE}/ae_ckpt
    ln -sfn $BASE/autoencoder/ckpt/${SCENE}_exp/best_ckpt.pth $CKPTS/${SCENE}/ae_ckpt/best_ckpt.pth

    cd $BASE/eval
    $PYTHON evaluate_3dovs.py --dataset_name $SCENE --dataset_path $SCENE_PATH \
        --feat_dir $OUT/3d_ovs --ae_ckpt_dir $CKPTS --output_dir $EVAL_OUT/3d_ovs --mask_thresh 0.4 \
        2>&1 | tee $SCENE_LOG/eval_${SCENE}.log
    cd $BASE

    EVAL_LOG=$(ls -t $EVAL_OUT/3d_ovs/$SCENE/*.log 2>/dev/null | head -1)
    MIOU=$(extract_miou_3dovs "${EVAL_LOG:-$SCENE_LOG/eval_${SCENE}.log}")

    # Update eval results - replace N/A line for this scene
    sed -i "s/^3D-OVS|${SCENE}|.*/3D-OVS|${SCENE}|${MIOU}/" $EVAL_RESULTS_FILE
    echo "[3D-OVS] $SCENE mIoU=$MIOU"
done

echo ""

# ============================================================
# Part 2: Re-run table scene (full pipeline)
# The symlinks for IMG_ language features are now in place
# ============================================================
echo "=========================================="
echo "Part 2: Re-running table scene"
echo "=========================================="

SCENE="table"
SCENE_PATH=$OVS_BASE/$SCENE
MODEL_BASE=$OUT/3d_ovs/$SCENE
SCENE_LOG=$LOG/3d_ovs
mkdir -p $SCENE_LOG

# AE already trained+tested, RGB 3DGS already trained (just LangSplat failed)
# We have the RGB checkpoint, skip AE and RGB steps

# Existing timing from logs
AE_TRAIN_TIME=$(extract_timing $SCENE_LOG/ae_train_${SCENE}.log "AE_TRAIN_TIME" 2>/dev/null || echo "0")
AE_TEST_TIME=$(extract_timing $SCENE_LOG/ae_test_${SCENE}.log "AE_TEST_TIME" 2>/dev/null || echo "0")
RECONSTRUCTION_TIME=$(extract_timing $SCENE_LOG/rgb_${SCENE}.log "RECONSTRUCTION_TIME" 2>/dev/null || echo "0")

echo "[table] LangSplat Training (levels 1,2,3)"
LIFTING_TOTAL=0
for level in 1 2 3; do
    echo "[table] LangSplat level $level"
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

# Update timing results - replace table line
sed -i "s/^3D-OVS|table|.*/3D-OVS|table|0|${RECONSTRUCTION_TIME:-0}|${LIFTING_3D}|0|${AE_TRAIN_TIME:-0}|${TOTAL}/" $RESULTS_FILE

echo "[table] Rendering"
for level in 1 2 3; do
    $PYTHON $BASE/render.py -s $SCENE_PATH -m ${MODEL_BASE}_${level} \
        --include_feature --skip_test \
        2>&1 | tee $SCENE_LOG/render_${SCENE}_level${level}.log
done

echo "[table] Evaluation"
mkdir -p $CKPTS/${SCENE}/ae_ckpt
ln -sfn $BASE/autoencoder/ckpt/${SCENE}_exp/best_ckpt.pth $CKPTS/${SCENE}/ae_ckpt/best_ckpt.pth
cd $BASE/eval
$PYTHON evaluate_3dovs.py --dataset_name $SCENE --dataset_path $SCENE_PATH \
    --feat_dir $OUT/3d_ovs --ae_ckpt_dir $CKPTS --output_dir $EVAL_OUT/3d_ovs --mask_thresh 0.4 \
    2>&1 | tee $SCENE_LOG/eval_${SCENE}.log
cd $BASE
EVAL_LOG=$(ls -t $EVAL_OUT/3d_ovs/$SCENE/*.log 2>/dev/null | head -1)
MIOU=$(extract_miou_3dovs "${EVAL_LOG:-$SCENE_LOG/eval_${SCENE}.log}")
sed -i "s/^3D-OVS|table|.*/3D-OVS|table|${MIOU}/" $EVAL_RESULTS_FILE
echo "[table] mIoU=$MIOU DONE"

echo ""

# ============================================================
# Part 3: DL3DV pipeline (4 scenes, full)
# ============================================================
echo "=========================================="
echo "Part 3: DL3DV Pipeline"
echo "=========================================="

for SCENE in furniture_shop office park_bench_car road_car_building; do
    SCENE_PATH=$DL3DV_BASE/$SCENE
    MODEL_BASE=$OUT/dl3dv/$SCENE
    SCENE_LOG=$LOG/dl3dv
    mkdir -p $SCENE_LOG

    echo ""
    echo "=========================================="
    echo "[DL3DV] Processing scene: $SCENE"
    echo "=========================================="

    # Step 1: Preprocess (SAM + CLIP)
    echo "[${SCENE}] Step 1: Preprocess (SAM + CLIP)"
    $PYTHON $BASE/preprocess.py --dataset_path $SCENE_PATH \
        2>&1 | tee $SCENE_LOG/preprocess_${SCENE}.log
    SAM_TIME=$(extract_timing $SCENE_LOG/preprocess_${SCENE}.log "SAM_TIME")
    CLIP_TIME=$(extract_timing $SCENE_LOG/preprocess_${SCENE}.log "CLIP_TIME")

    # Step 2: AE Train
    echo "[${SCENE}] Step 2: Autoencoder Training"
    cd $BASE/autoencoder
    $PYTHON train.py --dataset_path $SCENE_PATH --dataset_name ${SCENE}_exp \
        2>&1 | tee $SCENE_LOG/ae_train_${SCENE}.log
    cd $BASE
    AE_TRAIN_TIME=$(extract_timing $SCENE_LOG/ae_train_${SCENE}.log "AE_TRAIN_TIME")

    # Step 3: AE Test
    echo "[${SCENE}] Step 3: Autoencoder Test"
    cd $BASE/autoencoder
    $PYTHON test.py --dataset_path $SCENE_PATH --dataset_name ${SCENE}_exp \
        2>&1 | tee $SCENE_LOG/ae_test_${SCENE}.log
    cd $BASE
    AE_TEST_TIME=$(extract_timing $SCENE_LOG/ae_test_${SCENE}.log "AE_TEST_TIME")

    # Step 4: RGB 3DGS
    echo "[${SCENE}] Step 4: RGB 3DGS Training"
    $PYTHON $BASE/train.py -s $SCENE_PATH -m $MODEL_BASE --port $PORT \
        --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
        2>&1 | tee $SCENE_LOG/rgb_${SCENE}.log
    RECONSTRUCTION_TIME=$(extract_timing $SCENE_LOG/rgb_${SCENE}.log "RECONSTRUCTION_TIME")

    # Step 5: LangSplat x3
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

    # Step 6: Render
    echo "[${SCENE}] Step 6: Rendering"
    for level in 1 2 3; do
        $PYTHON $BASE/render.py -s $SCENE_PATH -m ${MODEL_BASE}_${level} \
            --include_feature --skip_test \
            2>&1 | tee $SCENE_LOG/render_${SCENE}_level${level}.log
    done

    # Step 7: Evaluate
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
done

echo ""
echo "===== All remaining work complete ====="
echo "End time: $(date)"
echo ""
echo "=== Final Timing Results ==="
cat $RESULTS_FILE
echo ""
echo "=== Final Eval Results ==="
cat $EVAL_RESULTS_FILE
