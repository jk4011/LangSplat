#!/usr/bin/env python
"""Evaluate 3D-OVS segmentation IoU (mIoU) for LangSplat."""
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm

import torch.nn.functional as F
import wandb

import sys
sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from utils import smooth, colormap_saving, vis_mask_save


def show_image_with_mask(target_img, mask, opacity=0.2):
    """Overlay mask on RGB image as RGBA. target_img: (3,H,W), mask: (H,W) or (1,H,W)."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if target_img.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask[None], size=target_img.shape[-2:], mode='nearest')[0]
    return torch.cat([target_img, (mask + opacity) / (1 + opacity)], dim=0)


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_3dovs(seg_folder: Path, image_folder: Path) -> Tuple[Dict, Tuple[int, int], List[str]]:
    """
    Load 3D-OVS ground truth annotations from segmentation masks.

    seg_folder: path to scene's segmentations/ directory
    image_folder: path to scene's images/ directory

    GT format: segmentations/{image_idx}/{class_name}.png
    Each PNG is a binary mask (0 or 255).

    Returns:
        gt_ann: dict mapping str(image_idx) -> {class_name: {'mask': np.array}}
        image_shape: (h, w)
        image_paths: list of image file paths for eval indices
    """
    # Read classes
    classes_file = seg_folder / 'classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    else:
        classes = None

    # Find all annotated image indices
    seg_dirs = sorted([d for d in seg_folder.iterdir() if d.is_dir()])

    gt_ann = {}
    image_shape = None
    eval_image_paths = []

    for seg_dir in seg_dirs:
        idx = int(seg_dir.name)
        img_ann = {}

        # Load all mask PNGs in this directory
        mask_files = sorted(seg_dir.glob('*.png'))
        for mask_file in mask_files:
            class_name = mask_file.stem  # filename without .png
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            # Convert to binary (0/1)
            mask = (mask > 127).astype(np.uint8)
            img_ann[class_name] = {'mask': mask}
            if image_shape is None:
                image_shape = mask.shape

        if img_ann:
            gt_ann[str(idx)] = img_ann

        # Find corresponding image (check multiple extensions)
        img_path = None
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            candidate = image_folder / f'{idx:02d}{ext}'
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            img_path = image_folder / f'{idx:02d}.png'  # fallback
        eval_image_paths.append(str(img_path))

    return gt_ann, image_shape, eval_image_paths


def compute_iou_3dovs(sem_map, image, clip_model, img_ann, image_name, thresh=0.5, colormap_options=None, wandb_images=None):
    """Compute per-class IoU for 3D-OVS format (no bboxes, just masks)."""
    valid_map = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    chosen_iou_list = []
    chosen_lvl_list = []

    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))

        for i in range(n_head):
            scale = 30
            kernel = np.ones((scale, scale)) / (scale ** 2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])

            if image_name is not None:
                output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
                output_path_relev.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options, output_path_relev)

            # Truncate heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * 2.0 - 1.0
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred

            # Resize GT mask to match prediction size if needed
            class_name = clip_model.positives[k]
            mask_gt = img_ann[class_name]['mask'].astype(np.uint8)
            if mask_gt.shape != (h, w):
                mask_gt = cv2.resize(mask_gt, (w, h), interpolation=cv2.INTER_NEAREST)

            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = intersection / (union + 1e-8)
            iou_lvl[i] = iou

        # Choose best level by max activation score
        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score_lvl[i] = valid_map[i, k].max()
        chosen_lvl = torch.argmax(score_lvl)

        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())

        if image_name is not None:
            save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
            vis_mask_save(mask_lvl[chosen_lvl], save_path)

        # wandb logging
        if wandb_images is not None:
            prompt = clip_model.positives[k]
            chosen_mask_pred = mask_lvl[chosen_lvl]
            mask_gt_wb = img_ann[prompt]['mask'].astype(np.uint8)
            if mask_gt_wb.shape != (h, w):
                mask_gt_wb = cv2.resize(mask_gt_wb, (w, h), interpolation=cv2.INTER_NEAREST)
            iou_val = iou_lvl[chosen_lvl]
            img_h, img_w = image.shape[0], image.shape[1]
            quarter_h, quarter_w = img_h // 4, img_w // 4
            img_chw = image.permute(2, 0, 1).cpu()
            img_quarter = F.interpolate(img_chw[None], size=(quarter_h, quarter_w), mode='bilinear', align_corners=False)[0]
            mask_gt_quarter = F.interpolate(
                torch.from_numpy(mask_gt_wb.astype(np.float32))[None, None], size=(quarter_h, quarter_w), mode='nearest'
            )[0, 0]
            mask_pred_quarter = F.interpolate(
                torch.from_numpy(chosen_mask_pred.astype(np.float32))[None, None], size=(quarter_h, quarter_w), mode='nearest'
            )[0, 0]
            gt_overlay = show_image_with_mask(img_quarter, mask_gt_quarter)
            pred_overlay = show_image_with_mask(img_quarter, mask_pred_quarter)
            key = f"frame_{image_name.name}/{prompt}"
            wandb_images[f"gt/{key}"] = wandb.Image(gt_overlay.permute(1, 2, 0).numpy(), caption=f"GT | IoU: {iou_val:.3f}")
            wandb_images[f"pred/{key}"] = wandb.Image(pred_overlay.permute(1, 2, 0).numpy(), caption=f"Pred | IoU: {iou_val:.3f}")

    return chosen_iou_list, chosen_lvl_list


def evaluate(feat_dir, output_path, ae_ckpt_path, seg_folder, image_folder,
             mask_thresh, encoder_hidden_dims, decoder_hidden_dims, logger, wandb_images=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo", normalize=True, colormap_min=-1.0, colormap_max=1.0,
    )

    gt_ann, image_shape, image_paths = eval_gt_3dovs(Path(seg_folder), Path(image_folder))
    eval_index_list = sorted([int(idx) for idx in gt_ann.keys()])

    logger.info(f"Evaluating {len(eval_index_list)} images with GT annotations")
    logger.info(f"Image shape: {image_shape}")

    # Load compressed features for eval indices
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                                key=lambda fn: int(os.path.basename(fn).split(".npy")[0]))
        for j, idx in enumerate(eval_index_list):
            feat = np.load(feat_paths_lvl[idx])
            # Resize if needed
            if feat.shape[:2] != image_shape:
                feat = cv2.resize(feat, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
            compressed_sem_feats[i][j] = feat

    # Load autoencoder and CLIP
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    chosen_iou_all = []
    chosen_lvl_list = []

    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx:05d}'
        image_name.mkdir(exist_ok=True, parents=True)

        sem_feat = compressed_sem_feats[:, j, ...]

        # Downscale if image is too large (>2M pixels per level) to avoid OOM
        orig_h, orig_w = sem_feat.shape[1], sem_feat.shape[2]
        scale_factor = 1
        while (orig_h // scale_factor) * (orig_w // scale_factor) > 2 * 1024 * 1024:
            scale_factor *= 2
        if scale_factor > 1:
            new_h, new_w = orig_h // scale_factor, orig_w // scale_factor
            sem_feat_scaled = np.zeros((sem_feat.shape[0], new_h, new_w, sem_feat.shape[3]), dtype=sem_feat.dtype)
            for li in range(sem_feat.shape[0]):
                sem_feat_scaled[li] = cv2.resize(sem_feat[li], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            sem_feat = sem_feat_scaled
            eval_h, eval_w = new_h, new_w
        else:
            eval_h, eval_w = orig_h, orig_w

        sem_feat = torch.from_numpy(sem_feat).float().to(device)

        rgb_img = cv2.imread(image_paths[j])[..., ::-1]
        if rgb_img.shape[:2] != (eval_h, eval_w):
            rgb_img = cv2.resize(rgb_img, (eval_w, eval_h))
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            # Decode level by level to avoid OOM on high-res images
            restored_levels = []
            batch_size = 128 * 1024  # 128K elements per batch
            for li in range(lvl):
                level_feat = sem_feat[li].flatten(0, 1)  # (h*w, 3)
                if level_feat.shape[0] > batch_size:
                    decoded_parts = []
                    for bi in range(0, level_feat.shape[0], batch_size):
                        part = model.decode(level_feat[bi:bi+batch_size])
                        decoded_parts.append(part)
                    level_decoded = torch.cat(decoded_parts, dim=0)
                else:
                    level_decoded = model.decode(level_feat)
                restored_levels.append(level_decoded.view(h, w, -1))
                torch.cuda.empty_cache()
            restored_feat = torch.stack(restored_levels, dim=0)  # (lvl, h, w, 512)

        img_ann = gt_ann[str(idx)]
        clip_model.set_positives(list(img_ann.keys()))

        c_iou_list, c_lvl = compute_iou_3dovs(
            restored_feat, rgb_img, clip_model, img_ann, image_name,
            thresh=mask_thresh, colormap_options=colormap_options,
            wandb_images=wandb_images
        )
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

    mean_iou = sum(chosen_iou_all) / len(chosen_iou_all) if chosen_iou_all else 0
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"mIoU: {mean_iou:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    return mean_iou


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_everything(42)

    parser = ArgumentParser(description="3D-OVS evaluation")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to scene directory')
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument("--ae_ckpt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument('--encoder_dims', nargs='+', type=int, default=[256, 128, 64, 32, 3])
    parser.add_argument('--decoder_dims', nargs='+', type=int, default=[16, 32, 64, 128, 256, 256, 512])
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    feat_dir = [os.path.join(args.feat_dir, dataset_name + f"_{i}", "train/ours_None/renders_npy") for i in range(1, 4)]
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "ae_ckpt/best_ckpt.pth")
    seg_folder = os.path.join(args.dataset_path, "segmentations")
    image_folder = os.path.join(args.dataset_path, "images")

    output_path = os.path.join(args.output_dir or "eval_output_3dovs", dataset_name)
    os.makedirs(output_path, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(dataset_name, log_file=log_file, log_level=logging.INFO)

    wandb_images = None
    if args.wandb:
        wandb.init(project="3dovs", group="langsplat", name=f"[langsplat] {dataset_name}")
        wandb_images = {}

    mean_iou = evaluate(feat_dir, output_path, ae_ckpt_path, seg_folder, image_folder,
             args.mask_thresh, args.encoder_dims, args.decoder_dims, logger, wandb_images=wandb_images)

    if args.wandb and wandb_images:
        log_dict = {"mIoU": mean_iou}
        log_dict.update(wandb_images)
        wandb.log(log_dict)
        wandb.finish()
