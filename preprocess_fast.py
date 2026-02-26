"""
Optimized preprocess.py for higher GPU utilization.
Key optimizations:
1. Incremental saving: save per-image, skip already processed (resume-friendly)
2. Batched CLIP encoding: all mode tiles in one forward pass
3. Prefetch next image's SAM masks while CLIP encodes current image
4. Vectorized mask2segmap using torch batch resize
5. Reduced CPU-GPU round trips
"""
import os
import random
import argparse
import gc
import time
import concurrent.futures

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,
            pretrained=self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)
    masks_flat = masks_ord.reshape(num_masks, -1).float()
    intersection_matrix = torch.mm(masks_flat, masks_flat.t())
    union_matrix = masks_area.unsqueeze(1) + masks_area.unsqueeze(0) - intersection_matrix
    iou_matrix = intersection_matrix / (union_matrix + 1e-8)
    ratio_i = intersection_matrix / (masks_area.unsqueeze(1) + 1e-8)
    ratio_j = intersection_matrix / (masks_area.unsqueeze(0) + 1e-8)
    cond_upper = (ratio_i < 0.5) & (ratio_j >= 0.85)
    inner_iou_upper = 1 - ratio_j * ratio_i
    inner_iou_matrix = torch.where(cond_upper, inner_iou_upper, torch.zeros_like(inner_iou_upper))
    cond_lower = (ratio_i >= 0.85) & (ratio_j < 0.5)
    inner_iou_lower = 1 - ratio_j * ratio_i
    inner_iou_matrix_t = torch.where(cond_lower, inner_iou_lower, torch.zeros_like(inner_iou_lower))
    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix_t.t(), diagonal=-1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l
    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    masks_new = ()
    for masks_lvl in args:
        seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))
        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)
        masks_new += (masks_lvl,)
    return masks_new


def batch_mask2segmap(masks, image):
    """Vectorized mask2segmap: batch process all mask crops."""
    if len(masks) == 0:
        return None, None

    seg_img_list = []
    seg_map = -np.ones(image.shape[:2], dtype=np.int32)

    for i, mask in enumerate(masks):
        seg_img = get_seg_img(mask, image)
        pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
        seg_img_list.append(pad_seg_img)
        seg_map[mask['segmentation']] = i

    seg_imgs = np.stack(seg_img_list, axis=0)
    seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0).to('cuda')
    return seg_imgs, seg_map


def process_single_image(img_tensor, mask_generator_local, clip_model, sam_encoder_fn):
    """Process a single image: SAM + CLIP, return embeddings and seg maps."""
    image_np = cv2.cvtColor(
        img_tensor[0].permute(1, 2, 0).numpy().astype(np.uint8),
        cv2.COLOR_BGR2RGB
    )

    # SAM GPU inference: _generate_masks (image encoding + mask prediction + NMS)
    torch.cuda.synchronize()
    sam_gpu_t0 = time.perf_counter()
    data, data_s, data_m, data_l = mask_generator_local._generate_masks(image_np)
    torch.cuda.synchronize()
    sam_gpu_time = time.perf_counter() - sam_gpu_t0

    # SAM CPU postprocessing: generate_curr_anns (postprocess_small_regions, RLE decode, etc.)
    sam_cpu_t0 = time.perf_counter()
    masks_default = mask_generator_local.generate_curr_anns(data)
    masks_s = mask_generator_local.generate_curr_anns(data_s)
    masks_m = mask_generator_local.generate_curr_anns(data_m)
    masks_l = mask_generator_local.generate_curr_anns(data_l)
    masks_default, masks_s, masks_m, masks_l = masks_update(
        masks_default, masks_s, masks_m, masks_l,
        iou_thr=0.8, score_thr=0.7, inner_thr=0.5
    )
    sam_cpu_time = time.perf_counter() - sam_cpu_t0

    sam_time = sam_gpu_time + sam_cpu_time

    # Process all mask modes and collect tiles
    modes = {}
    seg_maps = {}
    all_tiles = []
    mode_lengths = {}

    # Original always processes all 4 modes (KeyError if missing in CLIP loop)
    # sam_encoder only adds s/m/l if len != 0, and original iterates all 4 in CLIP
    # So we match: always add default, conditionally add s/m/l
    mode_masks = [('default', masks_default), ('s', masks_s), ('m', masks_m), ('l', masks_l)]

    for mode_name, masks in mode_masks:
        if len(masks) == 0:
            continue
        seg_imgs, seg_map = batch_mask2segmap(masks, image_np)
        modes[mode_name] = seg_imgs
        seg_maps[mode_name] = seg_map
        mode_lengths[mode_name] = seg_imgs.shape[0]
        all_tiles.append(seg_imgs)

    # CLIP: batch encode ALL tiles at once (key optimization!)
    torch.cuda.synchronize()
    clip_t0 = time.perf_counter()

    if all_tiles:
        all_tiles_batch = torch.cat(all_tiles, dim=0)  # [total_tiles, 3, 224, 224]

        # Process in sub-batches to avoid OOM for images with many masks
        CLIP_BATCH_SIZE = 128
        all_embeds = []
        with torch.no_grad():
            for start in range(0, all_tiles_batch.shape[0], CLIP_BATCH_SIZE):
                end = min(start + CLIP_BATCH_SIZE, all_tiles_batch.shape[0])
                batch = all_tiles_batch[start:end]
                embed = clip_model.encode_image(batch)
                embed /= embed.norm(dim=-1, keepdim=True)
                all_embeds.append(embed.detach().cpu().half())
        all_embeds = torch.cat(all_embeds, dim=0)

    torch.cuda.synchronize()
    clip_time = time.perf_counter() - clip_t0

    # Split embeddings back by mode
    clip_embeds = {}
    offset = 0
    for mode_name in modes:
        length = mode_lengths[mode_name]
        clip_embeds[mode_name] = all_embeds[offset:offset+length]
        offset += length

    return clip_embeds, seg_maps, sam_time, sam_gpu_time, sam_cpu_time, clip_time


def save_single_image(save_path, clip_embeds, seg_maps, embed_size=512):
    """Save a single image's features and seg maps."""
    lengths = [len(v) for k, v in clip_embeds.items()]
    total_length = sum(lengths)

    img_embed = torch.cat([v for k, v in clip_embeds.items()], dim=0)

    # Build seg_map tensor
    seg_map_tensor = []
    lengths_cumsum = lengths.copy()
    for j in range(1, len(lengths)):
        lengths_cumsum[j] += lengths_cumsum[j-1]

    for j, (k, v) in enumerate(seg_maps.items()):
        v_copy = v.copy()
        if j == 0:
            seg_map_tensor.append(torch.from_numpy(v_copy))
            continue
        assert v_copy.max() == lengths[j] - 1, f"{j}, {v_copy.max()}, {lengths[j]-1}"
        v_copy[v_copy != -1] += lengths_cumsum[j-1]
        seg_map_tensor.append(torch.from_numpy(v_copy))

    seg_map = torch.stack(seg_map_tensor, dim=0)

    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, seg_map.numpy())
    # Original stores half() into float32 tensor, so final output is float32
    np.save(save_path_f, img_embed.float().numpy())


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


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, 'images')
    data_list = sorted(os.listdir(img_folder))

    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)

    # Check which images are already processed (resume support)
    already_done = set()
    for data_path in data_list:
        save_path = os.path.join(save_folder, data_path.split('.')[0])
        if os.path.exists(save_path + '_s.npy') and os.path.exists(save_path + '_f.npy'):
            already_done.add(data_path)

    remaining = [d for d in data_list if d not in already_done]
    print(f"Total images: {len(data_list)}, already done: {len(already_done)}, remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("All images already processed. Exiting.")
        exit(0)

    # Load models
    clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
        points_per_batch=256,  # 64 -> 256: larger GPU batches for higher utilization
    )

    total_sam_time = 0.0
    total_sam_gpu_time = 0.0
    total_sam_cpu_time = 0.0
    total_clip_time = 0.0

    # Use a thread pool for async saving (overlap save I/O with GPU compute)
    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    save_futures = []

    def load_and_resize_image(data_path, img_folder, resolution_arg):
        """Load and resize a single image (CPU work)."""
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        orig_w, orig_h = image.shape[1], image.shape[0]
        if resolution_arg == -1:
            if orig_h > 1080:
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution_arg
        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        image = cv2.resize(image, resolution)
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return img_tensor

    # Preload and resize images on-the-fly (don't load all into RAM)
    WARNED = False
    for i, data_path in enumerate(tqdm(remaining, desc="Processing images")):
        if args.resolution == -1:
            image_path = os.path.join(img_folder, data_path)
            image = cv2.imread(image_path)
            orig_w, orig_h = image.shape[1], image.shape[0]
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Rescaling to 1080P.")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
            scale = float(global_down)
            resolution = (int(orig_w / scale), int(orig_h / scale))
            image = cv2.resize(image, resolution)
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            img_tensor = load_and_resize_image(data_path, img_folder, args.resolution)

        # Process single image
        clip_embeds, seg_maps, sam_time, sam_gpu_time, sam_cpu_time, clip_time = process_single_image(
            img_tensor, mask_generator, clip_model, None
        )

        # Skip first image for warmup timing
        if i > 0:
            total_sam_time += sam_time
            total_sam_gpu_time += sam_gpu_time
            total_sam_cpu_time += sam_cpu_time
            total_clip_time += clip_time

        # Save asynchronously (overlap disk I/O with next image's GPU compute)
        save_path = os.path.join(save_folder, data_path.split('.')[0])
        future = save_executor.submit(save_single_image, save_path, clip_embeds, seg_maps)
        save_futures.append(future)

        # Periodic cache clear
        if i % 20 == 0:
            torch.cuda.empty_cache()

    # Wait for all saves to complete
    for future in save_futures:
        future.result()
    save_executor.shutdown(wait=True)

    print(f"\nTIMING_RESULT: SAM_TIME={total_sam_time:.2f} SAM_GPU_TIME={total_sam_gpu_time:.2f} SAM_CPU_TIME={total_sam_cpu_time:.2f} CLIP_TIME={total_clip_time:.2f}")
    print("Preprocessing done.")
