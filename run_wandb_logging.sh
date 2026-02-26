#!/bin/bash
set -eo pipefail

export CUDA_VISIBLE_DEVICES=4

PYTHON=/opt/conda/envs/langsplat/bin/python
BASE=/root/data1/jinhyeok/LangSplat

# === Dataset paths ===
DL3DV_BASE=/root/data1/jinhyeok/seg123/dataset/dl3dv
DL3DV_LABEL=$DL3DV_BASE/label
OVSS_BASE=/root/data1/jinhyeok/seg123/dataset/3d_ovs

OUT=$BASE/output_experiment
EVAL_OUT=$BASE/eval_output_experiment
CKPTS=$BASE/ckpts_experiment

# ==========================================
# DL3DV - wandb logging (1/2 resolution, already default)
# ==========================================
echo "========== DL3DV wandb logging =========="

for SCENE in furniture_shop office park_bench_car road_car_building; do
    echo "[DL3DV] $SCENE - wandb eval"
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
        --wandb --wandb_project dl3dv
    cd $BASE
    echo "[DL3DV] $SCENE DONE"
done

# ==========================================
# 3D-OVS - wandb logging (1/4 resolution)
# ==========================================
echo "========== 3D-OVS wandb logging =========="

for SCENE in bed bench blue_sofa covered_desk lawn office_desk room snacks sofa table; do
    SCENE_PATH=$OVSS_BASE/$SCENE
    echo "[3D-OVS] $SCENE - wandb eval"
    mkdir -p $CKPTS/${SCENE}/ae_ckpt
    ln -sfn $BASE/autoencoder/ckpt/${SCENE}_exp/best_ckpt.pth $CKPTS/${SCENE}/ae_ckpt/best_ckpt.pth
    cd $BASE/eval
    $PYTHON evaluate_3dovs.py \
        --dataset_name $SCENE \
        --dataset_path $SCENE_PATH \
        --feat_dir $OUT/3d_ovs \
        --ae_ckpt_dir $CKPTS \
        --output_dir $EVAL_OUT/3d_ovs \
        --mask_thresh 0.5 \
        --wandb
    cd $BASE
    echo "[3D-OVS] $SCENE DONE"
done

echo ""
echo "===== All wandb logging complete ====="
