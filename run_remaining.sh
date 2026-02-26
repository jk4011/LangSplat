#!/bin/bash
# Run remaining steps: Preprocess -> Autoencoder -> LangSplat
# RGB 3DGS (Step 3) already completed

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
DATASET_BASE=$BASE/dataset/lerf_ovs
LOG=$BASE/log
mkdir -p $LOG

cd $BASE

SCENES="figurines ramen teatime waldo_kitchen"

# ===== Step 1: Preprocess (sequential, one scene at a time on GPU 6) =====
echo "===== Step 1: Preprocess ====="
for scene in $SCENES; do
    echo "[Preprocess] $scene on GPU 6"
    CUDA_VISIBLE_DEVICES=6 $PYTHON preprocess.py \
        --dataset_path $DATASET_BASE/$scene \
        2>&1 | tee $LOG/preprocess_${scene}.log
    if [ $? -ne 0 ]; then
        echo "ERROR: Preprocess failed for $scene"
        exit 1
    fi
    echo "[Preprocess] $scene done"
done

# ===== Step 2: Autoencoder Train + Test (sequential on GPU 6) =====
echo "===== Step 2: Autoencoder ====="
cd $BASE/autoencoder
for scene in $SCENES; do
    echo "[AE Train] $scene"
    CUDA_VISIBLE_DEVICES=6 $PYTHON train.py \
        --dataset_path $DATASET_BASE/$scene \
        --dataset_name $scene \
        2>&1 | tee $LOG/ae_train_${scene}.log
    if [ $? -ne 0 ]; then
        echo "ERROR: AE train failed for $scene"
        exit 1
    fi

    echo "[AE Test] $scene"
    CUDA_VISIBLE_DEVICES=6 $PYTHON test.py \
        --dataset_path $DATASET_BASE/$scene \
        --dataset_name $scene \
        2>&1 | tee $LOG/ae_test_${scene}.log
    if [ $? -ne 0 ]; then
        echo "ERROR: AE test failed for $scene"
        exit 1
    fi
    echo "[AE] $scene done"
done
cd $BASE

# ===== Step 4: LangSplat Training (2 scenes in parallel, 3 levels each) =====
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
