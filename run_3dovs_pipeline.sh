#!/bin/bash

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat
DATASET_BASE=$BASE/dataset/3d_ovs
LOG=$BASE/log/3dovs
mkdir -p $LOG
cd $BASE

SCENES="bed bench blue_sofa covered_desk lawn office_desk room snacks sofa table"
GPU1=1
GPU2=7

# ===== Step 1: Autoencoder Train + Test (2 at a time) =====
echo "===== Step 1: Autoencoder ====="
cd $BASE/autoencoder

scenes_arr=($SCENES)
for ((i=0; i<${#scenes_arr[@]}; i+=2)); do
    s1=${scenes_arr[$i]}
    s2=${scenes_arr[$((i+1))]}

    echo "[AE] $s1 (GPU$GPU1) + $s2 (GPU$GPU2)"
    (CUDA_VISIBLE_DEVICES=$GPU1 $PYTHON train.py --dataset_path $DATASET_BASE/$s1 --dataset_name $s1 \
        2>&1 | tee $LOG/ae_train_${s1}.log && \
     CUDA_VISIBLE_DEVICES=$GPU1 $PYTHON test.py --dataset_path $DATASET_BASE/$s1 --dataset_name $s1 \
        2>&1 | tee $LOG/ae_test_${s1}.log) &
    PID1=$!

    (CUDA_VISIBLE_DEVICES=$GPU2 $PYTHON train.py --dataset_path $DATASET_BASE/$s2 --dataset_name $s2 \
        2>&1 | tee $LOG/ae_train_${s2}.log && \
     CUDA_VISIBLE_DEVICES=$GPU2 $PYTHON test.py --dataset_path $DATASET_BASE/$s2 --dataset_name $s2 \
        2>&1 | tee $LOG/ae_test_${s2}.log) &
    PID2=$!

    wait $PID1 || { echo "FAILED: $s1 autoencoder"; exit 1; }
    wait $PID2 || { echo "FAILED: $s2 autoencoder"; exit 1; }
    echo "[AE] $s1 + $s2 done"
done

# Verify autoencoder output
echo "===== Verifying autoencoder output ====="
for scene in $SCENES; do
    count=$(ls $DATASET_BASE/$scene/language_features_dim3/*_s.npy 2>/dev/null | wc -l)
    echo "$scene: $count dim3 feature files"
    if [ "$count" -eq 0 ]; then
        echo "ERROR: No dim3 features for $scene!"
        exit 1
    fi
done

cd $BASE

# ===== Step 2: RGB 3DGS Training (for scenes without checkpoint) =====
echo "===== Step 2: RGB 3DGS Training ====="
NEED_RGB=""
for scene in $SCENES; do
    if [ ! -f "$DATASET_BASE/$scene/output_-1/chkpnt30000.pth" ]; then
        NEED_RGB="$NEED_RGB $scene"
    fi
done
echo "Scenes needing RGB 3DGS: $NEED_RGB"

rgb_arr=($NEED_RGB)
PORT=55570
for ((i=0; i<${#rgb_arr[@]}; i+=2)); do
    s1=${rgb_arr[$i]}
    s2=${rgb_arr[$((i+1))]-""}

    if [ -n "$s1" ] && [ -n "$s2" ]; then
        echo "[RGB] $s1 (GPU$GPU1) + $s2 (GPU$GPU2)"
        CUDA_VISIBLE_DEVICES=$GPU1 $PYTHON train.py -s $DATASET_BASE/$s1 -m $DATASET_BASE/$s1/output_-1 \
            --port $((PORT+i)) --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
            2>&1 | tee $LOG/rgb_${s1}.log &
        PID1=$!
        CUDA_VISIBLE_DEVICES=$GPU2 $PYTHON train.py -s $DATASET_BASE/$s2 -m $DATASET_BASE/$s2/output_-1 \
            --port $((PORT+i+1)) --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
            2>&1 | tee $LOG/rgb_${s2}.log &
        PID2=$!
        wait $PID1 || { echo "FAILED: $s1 RGB"; exit 1; }
        wait $PID2 || { echo "FAILED: $s2 RGB"; exit 1; }
    elif [ -n "$s1" ]; then
        echo "[RGB] $s1 (GPU$GPU1)"
        CUDA_VISIBLE_DEVICES=$GPU1 $PYTHON train.py -s $DATASET_BASE/$s1 -m $DATASET_BASE/$s1/output_-1 \
            --port $((PORT+i)) --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
            2>&1 | tee $LOG/rgb_${s1}.log
    fi
    echo "[RGB] batch done"
done

# Verify RGB checkpoints
echo "===== Verifying RGB checkpoints ====="
for scene in $SCENES; do
    if [ -f "$DATASET_BASE/$scene/output_-1/chkpnt30000.pth" ]; then
        echo "$scene: OK"
    else
        echo "$scene: MISSING checkpoint!"
        exit 1
    fi
done

# ===== Step 3: LangSplat Training (2 scenes in parallel) =====
echo "===== Step 3: LangSplat Training ====="
PORT=55580
for ((i=0; i<${#scenes_arr[@]}; i+=2)); do
    s1=${scenes_arr[$i]}
    s2=${scenes_arr[$((i+1))]}

    echo "[LangSplat] $s1 (GPU$GPU1) + $s2 (GPU$GPU2)"
    (for level in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$GPU1 $PYTHON train.py -s $DATASET_BASE/$s1 -m $DATASET_BASE/$s1/output_${level} \
            --start_checkpoint $DATASET_BASE/$s1/output_-1/chkpnt30000.pth \
            --feature_level $level --include_feature \
            --port $((PORT+i)) --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
            2>&1 | tee $LOG/langsplat_${s1}_level${level}.log
    done) &
    PID1=$!

    (for level in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$GPU2 $PYTHON train.py -s $DATASET_BASE/$s2 -m $DATASET_BASE/$s2/output_${level} \
            --start_checkpoint $DATASET_BASE/$s2/output_-1/chkpnt30000.pth \
            --feature_level $level --include_feature \
            --port $((PORT+i+1)) --checkpoint_iterations 30000 --test_iterations 30000 --save_iterations 30000 \
            2>&1 | tee $LOG/langsplat_${s2}_level${level}.log
    done) &
    PID2=$!

    wait $PID1 || { echo "FAILED: $s1 LangSplat"; exit 1; }
    wait $PID2 || { echo "FAILED: $s2 LangSplat"; exit 1; }
    echo "[LangSplat] $s1 + $s2 done"
done

# ===== Step 4: Render =====
echo "===== Step 4: Rendering ====="
for scene in $SCENES; do
    for level in 1 2 3; do
        echo "[Render] $scene level$level"
        CUDA_VISIBLE_DEVICES=2 $PYTHON render.py \
            -s $DATASET_BASE/$scene \
            -m $DATASET_BASE/$scene/output_${level} \
            --include_feature --skip_test \
            2>&1 | tee $LOG/render_${scene}_level${level}.log
    done
done

# ===== Step 5: Setup eval symlinks + Evaluate =====
echo "===== Step 5: Evaluation ====="
mkdir -p $BASE/output_3dovs
for scene in $SCENES; do
    for level in 1 2 3; do
        ln -sfn $DATASET_BASE/$scene/output_${level} $BASE/output_3dovs/${scene}_${level}
    done
    mkdir -p $BASE/ckpts/$scene/ae_ckpt
    ln -sfn $BASE/autoencoder/ckpt/$scene/best_ckpt.pth $BASE/ckpts/$scene/ae_ckpt/best_ckpt.pth
done

cd $BASE/eval
for scene in $SCENES; do
    echo "[Eval] $scene"
    CUDA_VISIBLE_DEVICES=2 $PYTHON evaluate_3dovs.py \
        --dataset_name $scene \
        --dataset_path $DATASET_BASE/$scene \
        --feat_dir $BASE/output_3dovs \
        --ae_ckpt_dir $BASE/ckpts \
        --output_dir $BASE/eval_output_3dovs \
        --mask_thresh 0.4 \
        2>&1 | tee $LOG/eval_${scene}.log
done

echo "===== All Done ====="
