#!/bin/bash
set -e

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
DATASET_BASE=$BASE/dataset/lerf_ovs
LOG=$BASE/log

mkdir -p $LOG
cd $BASE

# Note: train.py appends "_{feature_level}" to model_path.
# RGB 3DGS uses default feature_level=-1, so output goes to <model_path>_-1/
# LangSplat uses feature_level=1,2,3, so output goes to <model_path>_1/, _2/, _3/

# ===== Step 3: RGB 3DGS Training (2 scenes in parallel) =====
echo "===== Step 3: RGB 3DGS Training ====="

echo "[RGB 3DGS] figurines (GPU6) + ramen (GPU7)"
CUDA_VISIBLE_DEVICES=6 $PYTHON train.py -s $DATASET_BASE/figurines -m $DATASET_BASE/figurines/gs \
    --port 55556 --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
    2>&1 | tee $LOG/3dgs_rgb_figurines.log &
PID1=$!
CUDA_VISIBLE_DEVICES=7 $PYTHON train.py -s $DATASET_BASE/ramen -m $DATASET_BASE/ramen/gs \
    --port 55557 --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
    2>&1 | tee $LOG/3dgs_rgb_ramen.log &
PID2=$!
wait $PID1 $PID2

echo "[RGB 3DGS] teatime (GPU6) + waldo_kitchen (GPU7)"
CUDA_VISIBLE_DEVICES=6 $PYTHON train.py -s $DATASET_BASE/teatime -m $DATASET_BASE/teatime/gs \
    --port 55558 --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
    2>&1 | tee $LOG/3dgs_rgb_teatime.log &
PID1=$!
CUDA_VISIBLE_DEVICES=7 $PYTHON train.py -s $DATASET_BASE/waldo_kitchen -m $DATASET_BASE/waldo_kitchen/gs \
    --port 55559 --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
    2>&1 | tee $LOG/3dgs_rgb_waldo_kitchen.log &
PID2=$!
wait $PID1 $PID2

# ===== Step 4: LangSplat Training (2 scenes in parallel, 3 levels each) =====
# RGB checkpoint is at gs_-1/chkpnt30000.pth (because default feature_level=-1)
echo "===== Step 4: LangSplat Training ====="

echo "[LangSplat] figurines (GPU6) + ramen (GPU7)"
(for level in 1 2 3; do
    CUDA_VISIBLE_DEVICES=6 $PYTHON train.py -s $DATASET_BASE/figurines -m $DATASET_BASE/figurines/gs \
        --start_checkpoint $DATASET_BASE/figurines/gs_-1/chkpnt30000.pth \
        --feature_level $level --include_feature \
        --port 55560 --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
        2>&1 | tee $LOG/langsplat_figurines_level${level}.log
done) &
PID1=$!

(for level in 1 2 3; do
    CUDA_VISIBLE_DEVICES=7 $PYTHON train.py -s $DATASET_BASE/ramen -m $DATASET_BASE/ramen/gs \
        --start_checkpoint $DATASET_BASE/ramen/gs_-1/chkpnt30000.pth \
        --feature_level $level --include_feature \
        --port 55561 --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
        2>&1 | tee $LOG/langsplat_ramen_level${level}.log
done) &
PID2=$!
wait $PID1 $PID2

echo "[LangSplat] teatime (GPU6) + waldo_kitchen (GPU7)"
(for level in 1 2 3; do
    CUDA_VISIBLE_DEVICES=6 $PYTHON train.py -s $DATASET_BASE/teatime -m $DATASET_BASE/teatime/gs \
        --start_checkpoint $DATASET_BASE/teatime/gs_-1/chkpnt30000.pth \
        --feature_level $level --include_feature \
        --port 55562 --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
        2>&1 | tee $LOG/langsplat_teatime_level${level}.log
done) &
PID1=$!

(for level in 1 2 3; do
    CUDA_VISIBLE_DEVICES=7 $PYTHON train.py -s $DATASET_BASE/waldo_kitchen -m $DATASET_BASE/waldo_kitchen/gs \
        --start_checkpoint $DATASET_BASE/waldo_kitchen/gs_-1/chkpnt30000.pth \
        --feature_level $level --include_feature \
        --port 55563 --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
        2>&1 | tee $LOG/langsplat_waldo_kitchen_level${level}.log
done) &
PID2=$!
wait $PID1 $PID2

echo "===== All Done ====="
