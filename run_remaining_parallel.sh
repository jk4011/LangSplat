#!/bin/bash

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
DATASET_BASE=$BASE/dataset/lerf_ovs
LOG=$BASE/log
mkdir -p $LOG
cd $BASE

# ===== Step 1: Preprocess (parallel: 2 scenes at a time) =====
echo "===== Step 1: Preprocess (batch 1) ====="
echo "[Preprocess] figurines (GPU1) + ramen (GPU7)"
CUDA_VISIBLE_DEVICES=1 $PYTHON preprocess.py --dataset_path $DATASET_BASE/figurines \
    2>&1 | tee $LOG/preprocess_figurines.log &
PID1=$!
CUDA_VISIBLE_DEVICES=7 $PYTHON preprocess.py --dataset_path $DATASET_BASE/ramen \
    2>&1 | tee $LOG/preprocess_ramen.log &
PID2=$!
wait $PID1 $PID2
echo "[Preprocess] batch 1 done (exit codes: $?)"

echo "===== Step 1: Preprocess (batch 2) ====="
echo "[Preprocess] teatime (GPU1) + waldo_kitchen (GPU7)"
CUDA_VISIBLE_DEVICES=1 $PYTHON preprocess.py --dataset_path $DATASET_BASE/teatime \
    2>&1 | tee $LOG/preprocess_teatime.log &
PID1=$!
CUDA_VISIBLE_DEVICES=7 $PYTHON preprocess.py --dataset_path $DATASET_BASE/waldo_kitchen \
    2>&1 | tee $LOG/preprocess_waldo_kitchen.log &
PID2=$!
wait $PID1 $PID2
echo "[Preprocess] batch 2 done"

# Verify preprocess output
echo "===== Verifying preprocess output ====="
for scene in figurines ramen teatime waldo_kitchen; do
    count=$(ls $DATASET_BASE/$scene/language_features/*_f.npy 2>/dev/null | wc -l)
    echo "$scene: $count feature files"
    if [ "$count" -eq 0 ]; then
        echo "ERROR: No features for $scene!"
        exit 1
    fi
done

# ===== Step 2: Autoencoder Train + Test (parallel: 2 at a time on GPU 1,7) =====
echo "===== Step 2: Autoencoder (batch 1) ====="
cd $BASE/autoencoder

(CUDA_VISIBLE_DEVICES=1 $PYTHON train.py --dataset_path $DATASET_BASE/figurines --dataset_name figurines \
    2>&1 | tee $LOG/ae_train_figurines.log && \
 CUDA_VISIBLE_DEVICES=1 $PYTHON test.py --dataset_path $DATASET_BASE/figurines --dataset_name figurines \
    2>&1 | tee $LOG/ae_test_figurines.log) &
PID1=$!

(CUDA_VISIBLE_DEVICES=7 $PYTHON train.py --dataset_path $DATASET_BASE/ramen --dataset_name ramen \
    2>&1 | tee $LOG/ae_train_ramen.log && \
 CUDA_VISIBLE_DEVICES=7 $PYTHON test.py --dataset_path $DATASET_BASE/ramen --dataset_name ramen \
    2>&1 | tee $LOG/ae_test_ramen.log) &
PID2=$!
wait $PID1 $PID2
echo "[AE] batch 1 done"

echo "===== Step 2: Autoencoder (batch 2) ====="
(CUDA_VISIBLE_DEVICES=1 $PYTHON train.py --dataset_path $DATASET_BASE/teatime --dataset_name teatime \
    2>&1 | tee $LOG/ae_train_teatime.log && \
 CUDA_VISIBLE_DEVICES=1 $PYTHON test.py --dataset_path $DATASET_BASE/teatime --dataset_name teatime \
    2>&1 | tee $LOG/ae_test_teatime.log) &
PID1=$!

(CUDA_VISIBLE_DEVICES=7 $PYTHON train.py --dataset_path $DATASET_BASE/waldo_kitchen --dataset_name waldo_kitchen \
    2>&1 | tee $LOG/ae_train_waldo_kitchen.log && \
 CUDA_VISIBLE_DEVICES=7 $PYTHON test.py --dataset_path $DATASET_BASE/waldo_kitchen --dataset_name waldo_kitchen \
    2>&1 | tee $LOG/ae_test_waldo_kitchen.log) &
PID2=$!
wait $PID1 $PID2
echo "[AE] batch 2 done"

cd $BASE

# ===== Step 4: LangSplat Training (2 scenes in parallel, GPU 6,7) =====
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
