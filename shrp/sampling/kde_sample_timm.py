from collections import OrderedDict
from shrp.datasets.dataset_tokens_single import SingleModelTokens
from shrp.models.def_NN_experiment import NNmodule

import torch

from typing import Optional, List, Any

import numpy as np

from einops import repeat

from sklearn.neighbors import KernelDensity

import logging
from torch.utils.data import Subset

from torchmetrics.classification import JaccardIndex


import torchvision

import copy
from sklearn.metrics import average_precision_score

from shrp.sampling.condition_bn import condition_checkpoints, check_equivalence
from shrp.sampling.load_dataset import load_datasets_from_config
from shrp.sampling.evaluate_ensemble import evaluate_ensemble
from shrp.sampling.evaluate_single_model import (
    evaluate_single_model,
    evaluate_single_model_llm,
)
from shrp.sampling.de_normalize import de_normalize_checkpoint
from shrp.sampling.get_anchor_embeddings import (
    get_anchor_embeddings,
)
from shrp.sampling.sample_models import sample_models
from shrp.models.gpt_module import GPTModule

from pathlib import Path
import timm


import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from downstream_datasets.bigearthnet_data import BigEarthNetS2Dataset

# from downstream_datasets.datasets.dfc2020_data import DFC2020Dataset, s2_mean, s2_std
from downstream_datasets.dior_data import DIORDataset
from downstream_datasets.fmow_data import fMoWDataset
from downstream_datasets.eurosat_def import EuroSAT, TensorToPIL, PILToTensor, DictTransformWrapper
from downstream_datasets.resisc45_data import RESISC45Dataset
from downstream_datasets.spacenet_data import SpaceNetBuildingDataset
from downstream_datasets.sen1flood11_data import Sen1Floods11HandLabeledDataset
from downstream_datasets.sen12flood_data import SEN12FLOOD
from downstream_datasets.california_wildfires_data import CaliforniaWildfire

import albumentations as A
from albumentations.pytorch import ToTensorV2

import pandas as pd

import torch.nn.functional as F

from collections import OrderedDict


from timm.models.vision_transformer import resize_pos_embed

from shrp.models.vit_det import ViTDet, load_classifier_ckpt_into_vitdet
from shrp.models.resnet_det import ResNetDet, load_classifier_ckpt_into_resnetdet
from shrp.models.swin_det import SwinDet, load_classifier_ckpt_into_swinnet

# from geobench_loader import GeoBenchDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IGNORE = 255


@torch.no_grad()
def evaluate_detection_map(
    model, dataloader, num_classes, iou_thresh=0.5, use_11_point=False
):
    device = next(model.parameters()).device
    model.eval()

    all_dets, all_gts = [], []
    for images, targets in dataloader:
        images = [im.to(device) for im in images]
        outputs = model(images)  # inference: returns list of dicts

        for out, tgt in zip(outputs, targets):
            all_dets.append(
                {
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu(),
                }
            )
            all_gts.append(
                {
                    "boxes": tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu(),
                }
            )

    mAP, _ = evaluate_map_voc(
        all_dets,
        all_gts,
        num_classes=num_classes,
        iou_thresh=iou_thresh,
        use_11_point=use_11_point,
    )
    model.train()
    return mAP


def save_checkpoint(model, optimizer, path="checkpoints/vitdet.pth"):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    print(f"[checkpoint] saved: {checkpoint_path}")


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(
        inter_y2 - inter_y1, 0, None
    )
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def voc_ap(rec: np.ndarray, prec: np.ndarray, use_11_point: bool = False) -> float:
    if use_11_point:  # VOC07
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = prec[rec >= t].max() if np.any(rec >= t) else 0
            ap += p / 11.0
        return ap
    # VOC10/12 continuous interpolation
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map_voc(detections, gts, num_classes, iou_thresh=0.5, use_11_point=False):
    gt_by_class = {c: {} for c in range(1, num_classes)}
    npos = {c: 0 for c in range(1, num_classes)}

    for img_idx, gt in enumerate(gts):
        gtb = gt["boxes"].numpy().astype(np.float32)
        gtl = gt["labels"].numpy().astype(np.int64)
        for c in range(1, num_classes):
            mask = gtl == c
            boxes_c = gtb[mask]
            if boxes_c.size:
                gt_by_class[c][img_idx] = {
                    "boxes": boxes_c,
                    "detected": np.zeros((boxes_c.shape[0],), dtype=bool),
                }
                npos[c] += boxes_c.shape[0]

    aps = np.zeros((num_classes - 1,), dtype=np.float32)
    for c in range(1, num_classes):
        recs = []
        for img_idx, det in enumerate(detections):
            lab = det["labels"].numpy().astype(np.int64)
            m = lab == c
            if not np.any(m):
                continue
            boxes = det["boxes"].numpy().astype(np.float32)[m]
            scores = det["scores"].numpy().astype(np.float32)[m]
            for b, s in zip(boxes, scores):
                recs.append((img_idx, float(s), b))
        if not recs:
            aps[c - 1] = 0.0
            continue
        recs.sort(key=lambda x: -x[1])
        tp = np.zeros((len(recs),), dtype=np.float32)
        fp = np.zeros((len(recs),), dtype=np.float32)
        for i, (img_idx, score, box) in enumerate(recs):
            gtd = gt_by_class[c].get(img_idx)
            if gtd is None:
                fp[i] = 1.0
                continue
            ious = box_iou_np(box[None, :], gtd["boxes"]).squeeze(0)
            best = int(np.argmax(ious)) if ious.size else -1
            if best >= 0 and ious[best] >= iou_thresh and not gtd["detected"][best]:
                tp[i] = 1.0
                gtd["detected"][best] = True
            else:
                fp[i] = 1.0
        fp_c = np.cumsum(fp)
        tp_c = np.cumsum(tp)
        prec = tp_c / np.clip(tp_c + fp_c, 1e-9, None)
        rec = tp_c / max(npos[c], 1)
        aps[c - 1] = voc_ap(rec, prec, use_11_point=use_11_point)

    mAP = float(np.mean(aps)) if aps.size else 0.0
    return mAP, aps


def accumulate_binary_inter_union(preds, targets):
    """
    preds, targets: [B,H,W] in {0,1}
    Returns (inter_sum, union_sum, empty_count, B)
    """
    B = preds.size(0)
    inter = ((preds == 1) & (targets == 1)).view(B, -1).sum(dim=1).float()  # [B]
    union = ((preds == 1) | (targets == 1)).view(B, -1).sum(dim=1).float()  # [B]
    empty = union == 0
    return inter.sum().item(), union.sum().item(), int(empty.sum().item()), B


def finalize_binary_iou(inter_sum, union_sum, empty_count, sample_count, policy="skip"):
    """
    policy:
      - "skip": IoU over non-empty only (recommended)
      - "one" : count empty==empty as IoU=1, i.e., add empty_count to numerator & denominator
    """
    if policy == "skip":
        if union_sum == 0:
            return 0.0  # all empty; define as 0 (or 1) per your preference
        return inter_sum / max(union_sum, 1e-8)
    elif policy == "one":
        # empty-empty counts as 1.0
        return (inter_sum + empty_count) / max(union_sum + empty_count, 1e-8)
    else:
        raise ValueError("policy must be 'skip' or 'one'")


def binary_mean_iou(preds, targets, empty_policy="one"):
    """
    preds/targets: [B,H,W] int {0,1}
    empty_policy: "one" -> empty-empty counts as 1.0
                  "skip" -> ignore empty-empty samples
    """
    B = preds.size(0)
    inter = ((preds == 1) & (targets == 1)).view(B, -1).sum(dim=1).float()
    union = ((preds == 1) | (targets == 1)).view(B, -1).sum(dim=1).float()

    if empty_policy == "one":
        iou = torch.where(
            union == 0, torch.ones_like(union), inter / torch.clamp(union, min=1)
        )
        return iou.mean()
    elif empty_policy == "skip":
        keep = union > 0
        if keep.any():
            return (inter[keep] / union[keep]).mean()
        else:
            return torch.tensor(0.0, device=preds.device)  # or 1.0 if you prefer
    else:
        raise ValueError("empty_policy must be 'one' or 'skip'")


def _expand_conv2d_in_channels(old_w: torch.Tensor, in_ch: int, init_mode="avg"):
    """
    Expand pretrained conv weight [out, old_in, k, k] to [out, in_ch, k, k].
    """
    out_c, old_in, kh, kw = old_w.shape
    new_w = torch.zeros(out_c, in_ch, kh, kw, device=old_w.device, dtype=old_w.dtype)
    copy_c = min(old_in, in_ch)
    new_w[:, :copy_c] = old_w[:, :copy_c]
    if in_ch > old_in:
        extra = in_ch - old_in
        if init_mode == "avg":
            avg = old_w.mean(dim=1, keepdim=True)  # [out,1,k,k]
            new_w[:, old_in:] = avg.expand(-1, extra, -1, -1)
        elif init_mode == "repeat":
            reps = (extra + old_in - 1) // old_in
            tiled = old_w.repeat(1, reps, 1, 1)[:, :extra]
            new_w[:, old_in:] = tiled
        elif init_mode == "zero":
            pass
        else:
            raise ValueError(f"Unknown init_mode {init_mode}")
    return new_w


def remap_ckpt_to_timm(ckpt_state, model, choose_patch_idx=0, init_mode="avg"):
    """
    Remap your checkpoint (with keys like 'patch_embed.0.proj.*', 'blocks.*', 'decoder_*', etc.)
    to a timm ViT (with 'patch_embed.proj.*', 'blocks.*', etc.).
    """
    new_state = {}
    embed_dim = model.embed_dim
    in_chans = getattr(model.patch_embed.proj, "in_channels", 3)

    for k, v in ckpt_state.items():
        # skip decoder / MAE extras
        if k.startswith("decoder_") or k in {
            "mask_token",
            "decoder_pos_embed",
            "decoder_channel_embed",
            "channel_embed",
        }:
            continue

        if k == "cls_token":
            new_state["cls_token"] = v
            continue
        if k == "pos_embed":
            # We'll handle resizing below, but keep the raw tensor here
            new_state["pos_embed"] = v
            continue

        # patch embed: pick one index to map to timm's single embed
        if k.startswith("patch_embed."):
            parts = k.split(".")  # ['patch_embed', '{i}', 'proj', 'weight/bias']
            if len(parts) >= 4 and parts[2] == "proj":
                idx = int(parts[1])
                if idx == choose_patch_idx:
                    subkey = parts[-1]  # 'weight' or 'bias'
                    new_k = f"patch_embed.proj.{subkey}"
                    if subkey == "weight":
                        # shape check / expansion if needed
                        w = v
                        # Expand to match model's in_chans if needed
                        if w.shape[1] != in_chans:
                            w = _expand_conv2d_in_channels(
                                w, in_chans, init_mode=init_mode
                            )
                        # Match out channels / kernel automatically by shape check
                        if w.shape != model.patch_embed.proj.weight.shape:
                            raise ValueError(
                                f"Patch embed weight shape mismatch: "
                                f"{w.shape} vs {tuple(model.patch_embed.proj.weight.shape)}"
                            )
                        new_state[new_k] = w
                    else:
                        new_state[new_k] = v
                # else: ignore other patch_embed branches
            continue

        # transformer blocks: names already look timm-compatible (blocks.N.norm1, attn.qkv, etc.)
        if k.startswith("blocks."):
            new_state[k] = v
            continue

        # layernorm before head (timm uses 'norm.weight/bias')
        if k.startswith("norm."):
            new_state[k] = v
            continue

        # heads / head: ignore if you're going to set num_classes differently
        if k.startswith("head") or k.startswith("heads"):
            # you can keep it if shape matches; otherwise skip
            continue

        # Anything else is ignored by default
        # print("Ignoring key:", k)

    # ---- Positional embedding resize if needed ----
    if "pos_embed" in new_state:
        pe = new_state["pos_embed"]
        # timm pos_embed includes class token at index 0: [1, 1+N, C]
        if pe.ndim == 2:
            pe = pe.unsqueeze(0)
        if pe.shape[-1] != embed_dim:
            # sometimes different embed dims -> can't use
            print(
                f"[warn] pos_embed dim mismatch: ckpt {pe.shape[-1]} vs model {embed_dim}. Dropping."
            )
            new_state.pop("pos_embed", None)
        else:
            # use timm utility to resize grid (keeps cls token)
            new_state["pos_embed"] = resize_pos_embed(pe, model.pos_embed)

    return new_state


def safe_ce_loss(logits, targets, ignore_index=IGNORE):
    # logits: [B, C, H, W]; targets: [B, H, W] long
    assert targets.dtype == torch.long, f"targets dtype {targets.dtype}"
    assert torch.isfinite(logits).all(), "NaN/Inf in logits before loss"

    # Compute per-pixel loss while telling CE to ignore 255 labels
    per_pix = F.cross_entropy(
        logits, targets, reduction="none", ignore_index=ignore_index
    )  # now pixels with 255 contribute 0 and do not assert

    # Mask valid pixels explicitly to avoid NaN from averaging an all-ignored batch
    valid = targets != ignore_index
    if not valid.any():
        # return 0 with grad so the step is a no-op (or skip this batch)
        return logits.sum() * 0.0

    loss = per_pix[valid].mean()
    assert torch.isfinite(loss).all(), "NaN/Inf in loss after mask"
    return loss


def compute_mIoU(outputs, labels, num_classes=8, ignore_index=255):
    total_inter = torch.zeros(num_classes, device=outputs.device)
    total_union = torch.zeros(num_classes, device=outputs.device)

    for i in range(len(outputs)):
        if i == 0:
            print("outputs[i]", outputs[i])
        pred = torch.argmax(outputs[i], dim=0).flatten()
        label = labels[i].flatten()

        valid = label != ignore_index
        pred = pred[valid]
        label = label[valid]

        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = label == cls
            intersection = (pred_inds & target_inds).sum()
            union = (pred_inds | target_inds).sum()

            total_inter[cls] += intersection
            total_union[cls] += union

    valid_classes = total_union > 0
    iou_per_class = (
        total_inter[valid_classes].float() / total_union[valid_classes].float()
    )
    print(iou_per_class)
    print(valid_classes)
    mean_iou = iou_per_class.mean().item()
    return mean_iou


def load_classifier_ckpt_into_swinseg(seg_model, ckpt):
    model_sd = seg_model.state_dict()
    remapped = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("head"):  # drop classification head
            continue
        nk = f"backbone.{k}"  # prefix for the wrapped backbone
        if nk in model_sd and model_sd[nk].shape == v.shape:
            remapped[nk] = v
    # update and load
    model_sd.update(remapped)
    seg_model.load_state_dict(model_sd, strict=False)


def modify_backbone_input_channels(backbone, in_channels: int):
    # supports ViT/Swin from timm (features_only or not)
    pe = getattr(backbone, "patch_embed", None)
    proj = getattr(pe, "proj", None) if pe is not None else None
    if not isinstance(proj, torch.nn.Conv2d):
        raise RuntimeError("Backbone has no patch_embed.proj Conv2d")

    w = proj.weight  # [embed_dim, C_in, kh, kw]
    if w.shape[1] == in_channels:
        return backbone

    embed_dim, _, kh, kw = w.shape
    new_w = w.new_zeros((embed_dim, in_channels, kh, kw))
    # copy first 3 channels; init extras with RGB mean
    keep = min(3, in_channels, w.shape[1])
    new_w[:, :keep] = w[:, :keep]
    if in_channels > keep and w.shape[1] >= 3:
        mean_rgb = w[:, :3].mean(dim=1, keepdim=True)
        new_w[:, keep:] = mean_rgb.expand(embed_dim, in_channels - keep, kh, kw)

    proj.weight = torch.nn.Parameter(new_w)
    # bias unchanged
    return backbone


class SwinSeg(nn.Module):
    def __init__(self, model_name, num_classes, num_channels=3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=False, features_only=True
        )
        if num_channels != 3:
            self.backbone = modify_backbone_input_channels(
                self.backbone, in_channels=num_channels
            )

        channels = self.backbone.feature_info.channels()  # [96,192,384,768]
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, 128, kernel_size=1) for c in channels]
        )
        self.head = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)  # list of 4 features
        size = x.shape[-2:]
        upsampled = []

        for i, f in enumerate(feats):
            # If it's NHWC and matches expected in_channels on last dim, permute to NCHW
            if (
                f.ndim == 4
                and f.shape[-1] == self.lateral[i].in_channels
                and f.shape[1] != self.lateral[i].in_channels
            ):
                f = f.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW

            # Optional: if it's already NCHW but channels don't match, warn/raise
            # assert f.shape[1] == self.lateral[i].in_channels, f"Unexpected shape for feature {i}: {tuple(f.shape)}"

            f = self.lateral[i](f)  # [B, 128, h, w]
            f = F.interpolate(f, size=size, mode="bilinear", align_corners=False)
            upsampled.append(f)

        fused = torch.stack(upsampled, dim=0).sum(dim=0)
        return self.head(fused)  # [B, num_classes, H, W]


# def modify_vit_input_channels(
#     model: nn.Module, in_channels: int, fill_mode: str = "avg"
# ):
#     """
#     Modify a timm MobileNet (v2/v3) to accept `in_channels`.
#     It replaces the stem conv and sensibly initializes extra channels.
#     """
#     if not hasattr(model, "conv_stem"):
#         raise AttributeError(
#             "Model has no `conv_stem`. This helper targets timm MobileNet variants."
#         )

#     stem = model.conv_stem
#     old_conv = getattr(stem, "conv", stem)
#     if not isinstance(old_conv, nn.Conv2d):
#         raise TypeError("Expected a Conv2d at model.conv_stem or model.conv_stem.conv.")

#     # Cache properties + weights (detached)
#     W = old_conv.weight.detach().clone()
#     B = old_conv.bias.detach().clone() if old_conv.bias is not None else None
#     device, dtype = W.device, W.dtype

#     new_conv = nn.Conv2d(
#         in_channels=in_channels,
#         out_channels=old_conv.out_channels,
#         kernel_size=old_conv.kernel_size,
#         stride=old_conv.stride,
#         padding=old_conv.padding,
#         dilation=old_conv.dilation,
#         groups=1,  # MobileNet stem is not depthwise
#         bias=old_conv.bias is not None,
#         padding_mode=old_conv.padding_mode,
#     ).to(device=device, dtype=dtype)

#     # Initialize fresh
#     nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
#     if new_conv.bias is not None:
#         nn.init.zeros_(new_conv.bias)

#     cmin = min(old_conv.in_channels, in_channels)

#     # All weight/bias writes must happen under no_grad and via .copy_()
#     with torch.no_grad():
#         # Copy overlapping channels
#         new_conv.weight[:, :cmin, :, :].copy_(W[:, :cmin, :, :])

#         if in_channels > old_conv.in_channels:
#             if fill_mode == "avg":
#                 filler = W.mean(dim=1, keepdim=True)  # (out,1,k,k)
#                 repeat = in_channels - old_conv.in_channels
#                 new_conv.weight[:, old_conv.in_channels :, :, :].copy_(
#                     filler.expand(-1, repeat, -1, -1)
#                 )
#             elif fill_mode == "repeat":
#                 for c in range(old_conv.in_channels, in_channels):
#                     src = (c - old_conv.in_channels) % old_conv.in_channels
#                     new_conv.weight[:, c, :, :].copy_(W[:, src, :, :])
#             # else: already Kaiming-inited above

#         if new_conv.bias is not None and B is not None:
#             new_conv.bias.copy_(B)

#     # Swap into stem
#     if hasattr(stem, "conv"):
#         stem.conv = new_conv
#     else:
#         model.conv_stem = new_conv

#     # Update default_cfg input_size if present
#     if getattr(model, "default_cfg", None) and isinstance(model.default_cfg, dict):
#         cfg = dict(model.default_cfg)
#         if (
#             isinstance(cfg.get("input_size"), (list, tuple))
#             and len(cfg["input_size"]) == 3
#         ):
#             _, h, w = cfg["input_size"]
#             cfg["input_size"] = (in_channels, h, w)
#         model.default_cfg = cfg

#     return model


def modify_vit_input_channels(model, in_channels=13):
    """Modifies the input projection layer of a ViT model to accept `in_channels` input."""
    old_proj = model.patch_embed.proj
    model.patch_embed.proj = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None,
    )
    # You can also init weights if needed
    torch.nn.init.kaiming_normal_(
        model.patch_embed.proj.weight, mode="fan_out", nonlinearity="relu"
    )
    if model.patch_embed.proj.bias is not None:
        torch.nn.init.zeros_(model.patch_embed.proj.bias)
    return model


def set_num_classes(module: nn.Module, num_classes: int, task: str = "classification"):
    """
    Replace the classification head with a Linear that has `num_classes` outputs.
    Works with timm models (prefers reset_classifier) and generic PyTorch models.
    Skips for segmentation/detection tasks.
    """
    if task in {"segmentation", "detection"}:
        return module

    if hasattr(module, "reset_classifier") and callable(
        getattr(module, "reset_classifier")
    ):
        # keeps existing global_pool config
        module.reset_classifier(num_classes=num_classes)
        return module

    candidate_attrs = ["classifier", "head", "fc", "final_fc", "last_linear"]
    for attr in candidate_attrs:
        if hasattr(module, attr):
            layer = getattr(module, attr)
            # If it's a Linear already, reuse its in_features
            if isinstance(layer, nn.Linear):
                in_dim = layer.in_features
                setattr(module, attr, nn.Linear(in_dim, num_classes))
                return module
            # If it's Identity/None, we need to infer in_dim
            if layer is None or isinstance(layer, nn.Identity):
                in_dim = getattr(module, "num_features", None) or getattr(
                    module, "embed_dim", None
                )
                if in_dim is not None:
                    setattr(module, attr, nn.Linear(in_dim, num_classes))
                    return module
            # If it's a Sequential with last Linear, replace last Linear
            if (
                isinstance(layer, nn.Sequential)
                and len(layer) > 0
                and isinstance(layer[-1], nn.Linear)
            ):
                in_dim = layer[-1].in_features
                new_seq = nn.Sequential(*layer[:-1], nn.Linear(in_dim, num_classes))
                setattr(module, attr, new_seq)
                return module

    if hasattr(module, "get_classifier") and callable(
        getattr(module, "get_classifier")
    ):
        head = module.get_classifier()
        if isinstance(head, nn.Linear):
            in_dim = head.in_features
            # Prefer to attach back to a common attr name
            if hasattr(module, "classifier"):
                module.classifier = nn.Linear(in_dim, num_classes)
            elif hasattr(module, "head"):
                module.head = nn.Linear(in_dim, num_classes)
            else:
                # fall back: create `classifier`
                module.classifier = nn.Linear(in_dim, num_classes)
            return module

    in_dim = getattr(module, "num_features", None) or getattr(module, "embed_dim", None)
    if in_dim is not None:
        # Create a classifier attribute if none exists
        module.classifier = nn.Linear(in_dim, num_classes)
        return module

    raise AttributeError(
        "Could not locate a replaceable classification head or infer in_features. "
        "Provide a model-specific adapter or ensure the model exposes num_features/embed_dim."
    )


def sample_model_evaluation_timm(
    ae_model,
    finetuning_epochs: int,
    repetitions: int,
    reference_dataset_path: str,
    bootstrap_number: int,
    mode: str = "individual",  # 'individual','token,'joint'
    batch_size: int = 0,
    reset_classifier: bool = False,
    halo: bool = False,
    halo_wse: int = 156,
    halo_hs: int = 64,
    bn_condition_iters: int = 0,
    anchor_sample_number: int = 1,
    drop_samples_to_path: Optional[str | Path] = None,
    dense: bool = False,
    apply_layer_embs: bool = False,
    apply_layer_embs_enc_only: bool = False,
    use_relative_pos: bool = False,
    model_name: str = "resnet18",
    use_pretrained_anchor: bool = True,
    tokensize: int = 230,
    downstream_dataset: str = "eurosat",
    num_classes: int = 10,
    num_channels: int = 13,
    task: str = "singlelabel",
    linear_probing: bool = False,
) -> dict:
    """
    runs evaluation pipeline.
    samples hyper-representation model to generate checkpoints, finetunes checkpoints on downstream task and evaluates finetuned checkpoints
    Args:
        ae_model (hyper-representation): hyper-representation model
        sample_config (dict): dictionary containing config for sampled model
        finetuning_epochs (int): number of epochs to finetune
        repetitions (int): number of repetitions to finetune and evaluate
        anchor_ds_path: path to model dataset, which are used to fit the distributions to sample from
        reference_dataset_path: path to image dataset, which are used to evaluate the sampled models by for bootstrapping
        bootstrap_number: how many models to sample originally, among which the 'repetitions' are chosen.
        mode (str, optional): sampling mode. Defaults to "individual".
        norm_mode (Optional[str], optional): normalization mode. Defaults to None.
        layer_norms (Optional[dict], optional): normalization parameters. Defaults to None.
        batch_size (int, optional): batch size for evaluation. Defaults to 0.
        reset_classifier (bool, optional): whether to reset classifier. Defaults to True.
        halo (bool, optional): use halo-windows for encoding / decoding, instead of passing the entire sequence in one go. Defaults to False.
        halo_wse (int, optional): size of haloed-window. Defaults to 156.
        halo_hs (int, optional): size of the halo around the window. Defaults to 64.
        bn_condition_iters: (int, optional): if nonzero, perform conditioning iterations on train/val image dataset to tune bn statistics (only stats, no weight udpates)
        anchor_sample_number (int, optional): number of anchor samples to draw from anchor dataset. if 0, use all samples
    Returns:
        dict: dictionary containing evaluation results
    """
    assert bootstrap_number >= repetitions, (
        f"bootstrap number {bootstrap_number} needs to be larger than repetitions {repetitions}"
    )
    # init output
    results = {}
    # get reference model tokens
    logging.info("sampling:: create reference checkoint")

    if model_name == "satmae":
        module = timm.create_model(
            "vit_large_patch16_224", pretrained=False, patch_size=8
        )

        checkpoint_path = "satmae_checkpoint/pretrain-vit-large-e199.pth"
        raw = torch.load(checkpoint_path, map_location="cpu")
        state = raw.get("model", raw)

        remapped = remap_ckpt_to_timm(
            state, module, choose_patch_idx=0, init_mode="avg"
        )

        missing, unexpected = module.load_state_dict(remapped, strict=False)
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
    else:
        module = timm.create_model(
            model_name=model_name,
            pretrained=use_pretrained_anchor,
        )

    if num_channels != 3:
        module = modify_vit_input_channels(module, in_channels=num_channels)

    if not use_pretrained_anchor:
        train_loader, val_loader = get_data_loaders(
            data_root=reference_dataset_path,
            image_size=224,
            batch_size=32,  # 256
            num_workers=4,
        )
        # define criterion and optimizer
        if task == "multilabel":
            all_targets = []
            for _, y in train_loader:
                all_targets.append(y)
            all_targets = torch.cat(all_targets, dim=0)

            pos_counts = all_targets.sum(0)
            neg_counts = all_targets.size(0) - pos_counts
            pos_weight = neg_counts / (pos_counts + 1e-5)
            pos_weight = pos_weight.to(device)

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif task == "binary":
            criterion = nn.BCEWithLogitsLoss()
        elif task == "segmentation" and num_classes < 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=255)

        optimizer = optim.AdamW(module.parameters(), lr=5e-6)  # , weight_decay=1e-4)
        scaler = GradScaler(enabled=False)
        module.train()
        module.to("cuda")
        # finetune model
        for idx in range(1):
            train_loss, train_acc = train_one_epoch(
                module,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device="cuda",
                task=task,
                num_classes=num_classes,
            )
            val_loss, val_acc = evaluate(
                module,
                val_loader,
                criterion,
                device="cuda",
                task=task,
                num_classes=num_classes,
            )
            logging.info(
                f"model {model_name} finetuning: epoch {idx} train_loss: {train_loss} train_acc: {train_acc} val_loss: {val_loss} val_acc: {val_acc}"
            )
    dataset = SingleModelTokens(
        models=[module],
        tokensize=tokensize,
        ignore_bn=False,
        dense_tokens=dense,
        use_relative_pos=True,
    )
    checkpoint_ref = module.state_dict()
    anchor_dir = Path("anchor_tokenized")
    anchor_dir.mkdir(parents=True, exist_ok=True)
    anchor_ds_path = anchor_dir / f"{model_name}_dataset_tk_{tokensize}.pt"
    # save dataset to path, record path
    torch.save(dataset, anchor_ds_path)
    # get first anchor embeddings
    logging.info(f"sampling:: get first anchor embeddings from {anchor_ds_path}")
    anchor_z, anchor_pos, anchor_w_shape, anchor_types = get_anchor_embeddings(
        anchor_ds_path=str(anchor_ds_path),
        anchor_ds_ref=None,
        ae_model=ae_model,
        batch_size=batch_size,
        halo=halo,
        halo_wse=halo_wse,
        halo_hs=halo_hs,
        samples=anchor_sample_number,
        apply_layer_embs=apply_layer_embs,
        use_relative_pos=use_relative_pos,
    )

    # sample models
    logging.info("sampling:: sample models")
    checkpoints = sample_models(
        ae_model=ae_model,
        checkpoint_ref=checkpoint_ref,
        anchor_z=anchor_z,
        anchor_pos=anchor_pos,
        anchor_w_shape=anchor_w_shape,
        mode=mode,
        repetitions=bootstrap_number,
        batch_size=batch_size,
        return_new_anchor=False,
        reset_classifier=reset_classifier,
        halo=halo,
        halo_wse=halo_wse,
        halo_hs=halo_hs,
        dense=dense,
        anchor_types=anchor_types,
        apply_layer_embs=apply_layer_embs,
        apply_layer_embs_enc_only=apply_layer_embs_enc_only,
        use_relative_pos=use_relative_pos,
    )
    # cleanup
    del ae_model, checkpoint_ref, anchor_z, anchor_pos, anchor_w_shape, anchor_types

    # check if checkpoints are the same
    if check_equivalence(checkpoints[0], checkpoints[-1]):
        logging.warning(f"monitoring: same checkpoints after sampling")

    _, valloader = get_data_loaders(
        data_root=reference_dataset_path,
        image_size=800,
        batch_size=32,  # 512
        num_workers=4,
        dataset_name=downstream_dataset,
    )
    # condition bn of checkpoints
    checkpoints_new = []
    if bn_condition_iters > 0:
        for ckpt in checkpoints:
            # forward pass through model
            if model_name == "swin_s3_base_224.ms_in1k" and task != "detection":
                nn_model = SwinSeg(
                    "swin_s3_base_224.ms_in1k",
                    num_classes=num_classes,
                    num_channels=num_channels,
                )
                load_classifier_ckpt_into_swinseg(nn_model, ckpt)
            elif model_name == "satmae":
                nn_model = timm.create_model(
                    "vit_large_patch16_224", pretrained=False, patch_size=8
                )

                checkpoint_path = "satmae_checkpoint/pretrain-vit-large-e199.pth"
                raw = torch.load(checkpoint_path, map_location="cpu")
                state = raw.get("model", raw)

                remapped = remap_ckpt_to_timm(
                    state, nn_model, choose_patch_idx=0, init_mode="avg"
                )
                missing, unexpected = nn_model.load_state_dict(remapped, strict=False)

                if num_channels != 3:
                    nn_model = modify_vit_input_channels(
                        nn_model, in_channels=num_channels
                    )

            else:
                if task == "detection":
                    if "resnet" in model_name:
                        nn_model = ResNetDet(
                            model_name="resnet50",  # vit_large_patch16_224
                            num_classes=num_classes + 1,
                            num_channels=num_channels,
                            pretrained_backbone=False,
                        )
                        load_classifier_ckpt_into_resnetdet(nn_model, ckpt)
                    elif "swin" in model_name:
                        nn_model = SwinDet(
                            model_name="swin_s3_base_224.ms_in1k",
                            num_classes=num_classes + 1,
                            num_channels=num_channels,
                            pretrained_backbone=False,
                            img_size=(800, 800),
                        )
                        load_classifier_ckpt_into_swinnet(nn_model, ckpt)
                    else:
                        nn_model = ViTDet(
                            model_name="vit_large_patch16_224",
                            num_classes=num_classes + 1,
                            num_channels=num_channels,
                            pretrained_backbone=False,
                        )

                        load_classifier_ckpt_into_vitdet(nn_model, ckpt)
                else:
                    nn_model = timm.create_model(
                        model_name,
                        pretrained=False,
                    )

                    if num_channels != 3:
                        nn_model = modify_vit_input_channels(
                            nn_model, in_channels=num_channels
                        )

                    nn_model.load_state_dict(ckpt, strict=False)

            nn_model.train()
            nn_model.to("cuda")
            nn_model.eval()
            for idx, batch in enumerate(valloader):
                if idx > bn_condition_iters:
                    break
                imgx, _ = batch
                if task == "detection":
                    images = [im.to(device) for im in imgx]
                    imgx = images
                else:
                    imgx = imgx.to("cuda")
                # forward pass
                with torch.no_grad():
                    _ = nn_model(imgx)
            # set model back to eval mode
            nn_model.eval()
            nn_model.to("cpu")
            state_out = nn_model.state_dict()
            # replace checkpoint
            checkpoints_new.append(state_out)
    checkpoints = checkpoints_new

    # bootstrap: evaluate models on reference_dataset, keep  only the best ones
    logging.info("sampling:: timm model evaluation")
    checkpoints = timm_model_evaluation(
        checkpoints,
        keep_top_n=repetitions,
        reference_dataset_path=reference_dataset_path,
        model_name=model_name,
        num_channels=num_channels,
        num_classes=num_classes,
        downstream_dataset=downstream_dataset,
        task=task,
    )
    assert len(checkpoints) == repetitions, (
        f"after bootstrapping, checkpoints {len(checkpoints)} need to match repetitions {repetitions}"
    )

    # evaluate models
    logging.info("sampling:: evaluate models")
    for rep in range(repetitions):
        # sample model
        checkpoint = checkpoints[rep]
        # load model

        if model_name == "swin_s3_base_224.ms_in1k" and task != "detection":
            module = SwinSeg(
                "swin_s3_base_224.ms_in1k",
                num_classes=num_classes,
                num_channels=num_channels,
            )

        elif model_name == "satmae":
            module = timm.create_model(
                "vit_large_patch16_224", pretrained=False, patch_size=8
            )

            checkpoint_path = "satmae_checkpoint/pretrain-vit-large-e199.pth"
            raw = torch.load(checkpoint_path, map_location="cpu")
            state = raw.get("model", raw)

            remapped = remap_ckpt_to_timm(
                state, module, choose_patch_idx=0, init_mode="avg"
            )
            missing, unexpected = module.load_state_dict(remapped, strict=False)

            if num_channels != 3:
                module = modify_vit_input_channels(module, in_channels=num_channels)
        else:
            if task == "detection":
                if "resnet" in model_name:
                    module = ResNetDet(
                        model_name="resnet50",  # vit_large_patch16_224
                        num_classes=num_classes + 1,
                        num_channels=num_channels,
                        pretrained_backbone=False,
                    )
                elif "swin" in model_name:
                    module = SwinDet(
                        model_name="swin_s3_base_224.ms_in1k",
                        num_classes=num_classes + 1,
                        num_channels=num_channels,
                        pretrained_backbone=False,
                        img_size=(800, 800),
                    )
                else:
                    module = ViTDet(
                        model_name="vit_large_patch16_224",  # vit_large_patch16_224
                        num_classes=num_classes + 1,
                        num_channels=num_channels,
                        pretrained_backbone=False,
                    )
            else:
                module = timm.create_model(
                    model_name,
                    pretrained=False,
                )

                if num_channels != 3:
                    module = modify_vit_input_channels(module, in_channels=num_channels)

        # load checkpoint
        try:
            module.load_state_dict(checkpoint)
        except RuntimeError as e:
            logging.error(f"checkpoint loading failed: {e}")
            # probably wrong key naming
            key_new = ""
            key_old = "module."
            new_check = OrderedDict()
            for key in checkpoint.keys():
                nkey = key.replace(key_old, key_new)
                new_check[nkey] = checkpoint[key]
            module.load_state_dict(new_check)

        # if task != "segmentation" and task != "detection":
        #     if not hasattr(module, "head") or isinstance(module.head, nn.Identity):
        #         in_dim = getattr(module, "embed_dim", module.num_features)
        #         module.head = nn.Linear(in_dim, num_classes)
        #     else:
        #         in_dim = module.head.in_features
        #         module.head = nn.Linear(in_dim, num_classes)

        module = set_num_classes(module, num_classes=num_classes, task=task)

        module.eval()
        module.to("cuda")
        train_loader, val_loader = get_data_loaders(
            data_root=reference_dataset_path,
            image_size=800,
            batch_size=8,  # 512
            num_workers=8,
            dataset_name=downstream_dataset,
        )
        if task == "multilabel":
            all_targets = []
            for _, y in train_loader:
                all_targets.append(y)
            all_targets = torch.cat(all_targets, dim=0)

            pos_counts = all_targets.sum(0)
            neg_counts = all_targets.size(0) - pos_counts
            pos_weight = neg_counts / (pos_counts + 1e-5)
            pos_weight = pos_weight.to(device)

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif task == "segmentation":
            if num_classes > 1:
                criterion = nn.CrossEntropyLoss(ignore_index=255)
            else:
                criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(module.parameters(), lr=1e-5)  # , weight_decay=1e-4)
        scaler = GradScaler(enabled=False)

        res_tmp = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        # finetune model
        val_loss, val_acc = evaluate(
            module,
            val_loader,
            criterion,
            device="cuda",
            task=task,
            num_classes=num_classes,
        )
        res_tmp["val_loss_init"] = [val_loss]
        res_tmp["val_acc_init"] = [val_acc]
        if linear_probing:
            print("linear probing...")
            for param in module.parameters():
                param.requires_grad = False
            for param in module.head.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(module.head.parameters(), lr=1e-3)
        else:
            optimizer = optim.AdamW(module.parameters(), lr=1e-5)
        scaler = GradScaler(enabled=False)
        for idx in range(finetuning_epochs):
            train_loss, train_acc = train_one_epoch(
                module,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device="cuda",
                task=task,
                num_classes=num_classes,
            )
            val_loss, val_acc = evaluate(
                module,
                val_loader,
                criterion,
                device="cuda",
                task=task,
                num_classes=num_classes,
            )
            res_tmp["train_loss"].append(train_loss)
            res_tmp["train_acc"].append(train_acc)
            res_tmp["val_loss"].append(val_loss)
            res_tmp["val_acc"].append(val_acc)
            logging.info(
                f"model {rep} finetuning: epoch {idx} train_loss: {train_loss} train_acc: {train_acc} val_loss: {val_loss} val_acc: {val_acc}"
            )

        # append results
        for k in res_tmp.keys():
            results[f"model_{rep}_{k}"] = res_tmp[k]

        print(results)

        # # comparsion: model trained from scratch
        # # load model
        # res_tmp = {
        #     "train_loss": [],
        #     "train_acc": [],
        #     "val_loss": [],
        #     "val_acc": [],
        # }

        # if model_name == "swin_s3_base_224.ms_in1k" and task != "detection":
        #     module_scratch = SwinSeg(
        #         "swin_s3_base_224.ms_in1k",
        #         num_classes=num_classes,
        #         num_channels=num_channels,
        #     )
        # elif model_name == "satmae":
        #     module_scratch = timm.create_model(
        #         "vit_large_patch16_224", pretrained=False, patch_size=8
        #     )
        #     checkpoint_path = "satmae_checkpoint/pretrain-vit-large-e199.pth"
        #     raw = torch.load(checkpoint_path, map_location="cpu")
        #     state = raw.get("model", raw)

        #     remapped = remap_ckpt_to_timm(
        #         state, module_scratch, choose_patch_idx=0, init_mode="avg"
        #     )
        #     missing, unexpected = module_scratch.load_state_dict(remapped, strict=False)

        #     if num_channels != 3:
        #         module_scratch = modify_vit_input_channels(
        #             module_scratch, in_channels=num_channels
        #         )
        # else:
        #     if task == "detection":
        #         if "resnet" in model_name:
        #             module_scratch = ResNetDet(
        #                 model_name="resnet50",  # vit_large_patch16_224
        #                 num_classes=num_classes + 1,
        #                 num_channels=num_channels,
        #                 pretrained_backbone=False,
        #             )
        #         elif "swin" in model_name:
        #             module_scratch = SwinDet(
        #                 model_name="swin_s3_base_224.ms_in1k",
        #                 num_classes=num_classes + 1,
        #                 num_channels=num_channels,
        #                 pretrained_backbone=False,
        #                 img_size=(800, 800),
        #             )
        #         else:
        #             module_scratch = ViTDet(
        #                 model_name="vit_large_patch16_224",  # vit_large_patch16_224
        #                 num_classes=num_classes + 1,
        #                 num_channels=num_channels,
        #                 pretrained_backbone=False,
        #             )
        #     else:
        #         module_scratch = timm.create_model(
        #             model_name,
        #             pretrained=False,
        #         )
        #         if num_channels != 3:
        #             module_scratch = modify_vit_input_channels(
        #                 module_scratch, in_channels=num_channels
        #             )

        # # if task != "segmentation" and task != "detection":
        # #     if not hasattr(module_scratch, "head") or isinstance(
        # #         module_scratch.head, nn.Identity
        # #     ):
        # #         in_dim = getattr(
        # #             module_scratch, "embed_dim", module_scratch.num_features
        # #         )
        # #         module_scratch.head = nn.Linear(in_dim, num_classes)
        # #     else:
        # #         in_dim = module_scratch.head.in_features
        # #         module_scratch.head = nn.Linear(in_dim, num_classes)

        # module_scratch = set_num_classes(
        #     module_scratch, num_classes=num_classes, task=task
        # )

        # module_scratch.to("cuda")
        # # optimizer = optim.Adam(module_scratch.parameters(), lr=5e-5)
        # optimizer = optim.AdamW(
        #     module_scratch.parameters(), lr=1e-5
        # )  # , weight_decay=1e-4) #0.05

        # scaler = GradScaler(enabled=False)
        # val_loss, val_acc = evaluate(
        #     module_scratch,
        #     val_loader,
        #     criterion,
        #     device="cuda",
        #     task=task,
        #     num_classes=num_classes,
        # )
        # res_tmp["val_loss_init"] = [val_loss]
        # res_tmp["val_acc_init"] = [val_acc]
        # for idx in range(finetuning_epochs):
        #     train_loss, train_acc = train_one_epoch(
        #         module_scratch,
        #         train_loader,
        #         criterion,
        #         optimizer,
        #         scaler,
        #         device="cuda",
        #         task=task,
        #         num_classes=num_classes,
        #     )
        #     val_loss, val_acc = evaluate(
        #         module_scratch,
        #         val_loader,
        #         criterion,
        #         device="cuda",
        #         task=task,
        #         num_classes=num_classes,
        #     )
        #     res_tmp["train_loss"].append(train_loss)
        #     res_tmp["train_acc"].append(train_acc)
        #     res_tmp["val_loss"].append(val_loss)
        #     res_tmp["val_acc"].append(val_acc)
        #     logging.info(
        #         f"model {rep} finetuning: epoch {idx} train_loss: {train_loss} train_acc: {train_acc} val_loss: {val_loss} val_acc: {val_acc}"
        #     )

        # # append results
        # for k in res_tmp.keys():
        #     results[f"model_{rep}_{k}_scratch"] = res_tmp[k]

    # aggregate results over models
    for k in res_tmp.keys():
        res_tmp = []
        for rep in range(repetitions):
            res_tmp.append(results[f"model_{rep}_{k}"])

        results[f"{k}_mean"] = []
        results[f"{k}_std"] = []
        for idx in range(len(res_tmp[0])):
            res_ep = [res_tmp[jdx][idx] for jdx in range(len(res_tmp))]
            results[f"{k}_mean"].append(np.mean(res_ep))
            results[f"{k}_std"].append(np.std(res_ep))

    # return results
    return results


# def timm_model_evaluation(checkpoints, keep_top_n, model_name, reference_dataset_path):
#     """
#     evaluates a list of checkpoints on a single task
#     keeps only the top n checkpoints
#     Args:
#         checkpoints (list): list of state dicts of the model
#         keep_top_n (int): number of top checkpoints to keep
#         reference_dataset_path: path to image dataset, which are used to evaluate the sampled models by for bootstrapping
#     Returns:
#         list: list of top checkpoints
#     """
#     module = timm.create_model(
#         model_name,
#         pretrained=False
#     )
#     _, val_loader = get_data_loaders(
#         data_root=reference_dataset_path,
#         image_size=224,
#         batch_size=512,
#         num_workers=4,
#     )

#     criterion = nn.CrossEntropyLoss()
#     #
#     model_perf = []
#     #

#     in_features = module.head.in_features
#     module.head = nn.Linear(in_features, 10)
#     for idx in range(len(checkpoints)):
#         # get checkpoint
#         check = checkpoints[idx]
#         # load checkpoint
#         logging.info("load checkpoint model")
#         try:
#             # Get the model's current state_dict
#             model_dict = module.state_dict()
#             filtered_dict = {
#                     k: v for k, v in check.items()
#                     if k in model_dict and not k.startswith("head") and model_dict[k].shape == v.shape
#             }
#             # Load only matching weights
#             model_dict.update(filtered_dict)
#             module.load_state_dict(model_dict)

#             # module.load_state_dict(check)
#         except RuntimeError as e:
#             key_new = ""
#             key_old = "module."
#             new_check = OrderedDict()
#             for key in check.keys():
#                 nkey = key.replace(key_old, key_new)
#                 new_check[nkey] = check[key]
#             module.load_state_dict(new_check)


#         module.eval()
#         module.to("cuda")
#         # evaluate model
#         val_loss, val_acc = evaluate(
#             module,
#             val_loader,
#             criterion,
#             device="cuda",
#         )
#         # append model performance
#         model_perf.append((val_acc, idx))
#         logging.info(f"model {idx} performance: {val_acc}")

#     # sort model_perf by acc_val in descending order
#     model_perf = sorted(model_perf, key=lambda x: x[0], reverse=True)
#     assert model_perf[0][0] >= model_perf[-1][0], "model_perf not sorted correctly"

#     logging.info(f"subsampling: sorted model performance: {model_perf}")

#     # extract keep_top_n best models from model_perf
#     model_perf = model_perf[:keep_top_n]
#     # ectract indices of (acc,  ind) tuples
#     model_ind = [ddx[1] for ddx in model_perf]

#     # apply index on top_n  checkpoints
#     checkpoints_top_n = [checkpoints[idx] for idx in model_ind]

#     return checkpoints_top_n


def timm_model_evaluation(
    checkpoints,
    keep_top_n,
    model_name,
    reference_dataset_path,
    num_channels,
    num_classes,
    downstream_dataset,
    task,
):
    """
    evaluates a list of checkpoints on a single task
    keeps only the top n checkpoints
    """
    if model_name == "swin_s3_base_224.ms_in1k" and task != "detection":
        module = SwinSeg(
            "swin_s3_base_224.ms_in1k",
            num_classes=num_classes,
            num_channels=num_channels,
        )
    elif model_name == "satmae":
        module = timm.create_model(
            "vit_large_patch16_224", pretrained=False, patch_size=8
        )
        checkpoint_path = "satmae_checkpoint/pretrain-vit-large-e199.pth"
        raw = torch.load(checkpoint_path, map_location="cpu")
        state = raw.get("model", raw)

        remapped = remap_ckpt_to_timm(
            state, module, choose_patch_idx=0, init_mode="avg"
        )
        missing, unexpected = module.load_state_dict(remapped, strict=False)

        if num_channels != 3:
            module = modify_vit_input_channels(module, in_channels=num_channels)
    else:
        if task == "detection":
            if "resnet" in model_name:
                module = ResNetDet(
                    model_name="resnet50",  # vit_large_patch16_224
                    num_classes=num_classes + 1,
                    num_channels=num_channels,
                    pretrained_backbone=False,
                )
            elif "swin" in model_name:
                module = SwinDet(
                    model_name="swin_s3_base_224.ms_in1k",
                    num_classes=num_classes + 1,
                    num_channels=num_channels,
                    pretrained_backbone=False,
                    img_size=(800, 800),
                )
            else:
                module = ViTDet(
                    model_name="vit_large_patch16_224",
                    num_classes=num_classes + 1,
                    num_channels=num_channels,
                    pretrained_backbone=False,
                )
        else:
            module = timm.create_model(model_name, pretrained=False)
            if num_channels != 3:
                module = modify_vit_input_channels(module, in_channels=num_channels)

    # if task != "segmentation" and task != "detection":
    #     if not hasattr(module, "head") or isinstance(module.head, nn.Identity):
    #         in_dim = getattr(module, "embed_dim", module.num_features)
    #         module.head = nn.Linear(in_dim, num_classes)
    #     else:
    #         in_dim = module.head.in_features
    #         module.head = nn.Linear(in_dim, num_classes)

    module = set_num_classes(module, num_classes=num_classes, task=task)

    train_loader, val_loader = get_data_loaders(
        data_root=reference_dataset_path,
        image_size=800,
        batch_size=8,  # 512
        num_workers=4,
        dataset_name=downstream_dataset,
    )

    if task == "multilabel":
        all_targets = []
        for _, y in train_loader:
            all_targets.append(y)
        all_targets = torch.cat(all_targets, dim=0)

        pos_counts = all_targets.sum(0)
        neg_counts = all_targets.size(0) - pos_counts
        pos_weight = neg_counts / (pos_counts + 1e-5)
        pos_weight = pos_weight.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task == "segmentation":
        if num_classes < 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=255)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
    model_perf = []

    for idx in range(len(checkpoints)):
        check = checkpoints[idx]
        logging.info("load checkpoint model")
        try:
            model_dict = module.state_dict()
            filtered_dict = {
                k: v
                for k, v in check.items()
                if k in model_dict
                and not k.startswith("head")
                and model_dict[k].shape == v.shape
            }
            model_dict.update(filtered_dict)
            module.load_state_dict(model_dict)
        except RuntimeError as e:
            logging.warning(f"Fallback checkpoint loading for model {idx}: {e}")
            new_check = OrderedDict()
            for key in check.keys():
                nkey = key.replace("module.", "")
                new_check[nkey] = check[key]
            module.load_state_dict(new_check)

        module.eval()
        module.to("cuda")

        val_loss, val_acc = evaluate(
            module,
            val_loader,
            criterion,
            device="cuda",
            task=task,
            num_classes=num_classes,
        )
        model_perf.append((val_acc, idx))
        logging.info(f"model {idx} performance: {val_acc}")

    model_perf = sorted(model_perf, key=lambda x: x[0], reverse=True)
    logging.info(f"subsampling: sorted model performance: {model_perf}")
    model_perf = model_perf[:keep_top_n]
    model_ind = [ddx[1] for ddx in model_perf]
    checkpoints_top_n = [checkpoints[idx] for idx in model_ind]

    return checkpoints_top_n


# use this everywhere you build a detection DataLoader
def detection_collate(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


def get_data_loaders(data_root, image_size, batch_size, num_workers, dataset_name):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(image_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    # val_transform = transforms.Compose([
    #     transforms.Resize(int(image_size * 1.14)),
    #     transforms.CenterCrop(image_size),
    #     transforms.ToTensor(),
    #     normalize
    # ])

    # train_dataset = datasets.ImageNet(root=data_root, split='train', transform=train_transform)
    # val_dataset = datasets.ImageNet(root=data_root, split='val', transform=val_transform)

    if dataset_name == "eurosat":
        eurosat_path = "/netscratch1/lscheibenreif/code/low-rank-da/data"

        mean = [
            0.1393,
            0.1279,
            0.2641,
            0.2586,
            0.2904,
            0.2997,
            0.3263,
            0.3280,
            0.3265,
            0.0724,
            0.1542,
            0.1042,
            0.2285,
        ]
        std = [
            0.1076,
            0.0998,
            0.1362,
            0.1418,
            0.1431,
            0.1395,
            0.1449,
            0.1444,
            0.1435,
            0.0551,
            0.1021,
            0.0719,
            0.1335,
        ]

        train_transforms = DictTransformWrapper(
            A.Compose(
                [
                    A.Resize(224, 224),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5),
                    A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                    ToTensorV2(),
                ]
            )
        )

        test_transforms = DictTransformWrapper(
            A.Compose(
                [
                    A.Resize(224, 224),
                    A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                    ToTensorV2(),
                ]
            )
        )

        train_dataset = EuroSAT(
            root=eurosat_path,
            split="train",
            transforms=train_transforms,
        )

        val_dataset = EuroSAT(
            root=eurosat_path,
            split="val",
            transforms=test_transforms,
        )
    elif dataset_name == "resisc45":
        train_split = "datasets/resisc45-train.txt"
        val_split = "datasets/resisc45-val.txt"

        data_dir = "/ds2/remote_sensing/NWPU-RESISC45"

        train_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(20),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        train_dataset = RESISC45Dataset(
            data_dir=data_dir,
            split_file=train_split,
            transform=train_transforms,
        )

        val_dataset = RESISC45Dataset(
            data_dir=data_dir,
            split_file=val_split,
            transform=test_transforms,
        )
    elif dataset_name == "sen12flood":
        train_dataset = SEN12FLOOD(
            data_dir="/ds2/remote_sensing/sen12-flood/SEN12FLOOD",
            split="train",
            bands=["VH", "VV"],
            img_size=224,
        )
        val_dataset = SEN12FLOOD(
            data_dir="/ds2/remote_sensing/sen12-flood/SEN12FLOOD",
            split="val",
            bands=["VH", "VV"],
            img_size=224,
        )
    elif dataset_name == "california_fires":
        train_dataset = CaliforniaWildfire(
            data_dir="/ds2/remote_sensing/california_wildfires/images/",
            split="train",
            bands=["B11", "B12", "B8A"],
            img_size=224,
        )
        val_dataset = CaliforniaWildfire(
            data_dir="/ds2/remote_sensing/california_wildfires/images/",
            split="val",
            bands=["B11", "B12", "B8A"],
            img_size=224,
        )
    elif dataset_name == "fmow":
        data_dir = "/local/remote-sensing/fMoW"

        train_transforms = transforms.Compose(
            [
                transforms.Resize(120),
                transforms.CenterCrop(120),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.Resize(120),
                transforms.CenterCrop(120),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        trainset = fMoWDataset(
            root_dir=data_dir,
            split="training",
            transform=train_transforms,
        )

        testset = fMoWDataset(
            root_dir=data_dir,
            split="validation",
            transform=test_transforms,
        )

        train_n = int(0.1 * len(trainset))
        test_n = int(0.1 * len(testset))

        _, train_dataset = random_split(trainset, [len(trainset) - train_n, train_n])
        _, val_dataset = random_split(testset, [len(testset) - test_n, test_n])

    elif dataset_name == "bigearthnet":
        # S2_MEAN = [710.5539, 782.0168, 922.3734, 910.3503, 1261.4490, 2043.7069, 2292.5769, 2445.5312, 2460.3430, 2445.8997, 1490.5514, 962.4509, 962.4509]
        # S2_STD = [1436.4235, 1453.4495, 1342.0491, 1433.9960, 1433.9741, 1515.6384, 1571.7488, 1683.5098, 1590.1483, 1528.8700, 1053.8224, 784.3145, 784.3145]

        # Load your parquet
        df = pd.read_parquet("/ds2/remote_sensing/bigearthnet/metadata.parquet")

        # Build label map from full dataset
        all_labels = set(label for labels in df["labels"] for label in labels)
        label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}

        train_transforms = A.Compose(
            # [A.Normalize(mean = S2_MEAN, std = S2_STD),
            [ToTensorV2()]
        )

        test_transforms = A.Compose(
            # [A.Normalize(mean = S2_MEAN, std = S2_STD),
            [ToTensorV2()]
        )

        full_train_dataset = BigEarthNetS2Dataset(
            df=df,
            root_dir="/ds2/remote_sensing/bigearthnet/BigEarthNet-S2",
            split="train",
            label_map=label_map,
            transform=train_transforms,
        )

        full_val_dataset = BigEarthNetS2Dataset(
            df=df,
            root_dir="/ds2/remote_sensing/bigearthnet/BigEarthNet-S2",
            split="validation",
            label_map=label_map,
            transform=test_transforms,
        )

        train_n = int(0.1 * len(full_train_dataset))
        test_n = int(0.1 * len(full_val_dataset))

        _, train_dataset = random_split(
            full_train_dataset, [len(full_train_dataset) - train_n, train_n]
        )
        _, val_dataset = random_split(
            full_val_dataset, [len(full_val_dataset) - test_n, test_n]
        )

    elif dataset_name == "dfc":
        train_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # A.Normalize(mean=s2_mean, std=s2_std, max_pixel_value=10000),
                ToTensorV2(),
            ]
        )

        df = pd.read_csv("/ds2/remote_sensing/DFC2020/metadata.csv")

        train_dataset = DFC2020Dataset(
            dataframe=df,
            root_dir="/ds2/remote_sensing/DFC2020",
            split="train",
            transform=train_transform,
        )
        val_dataset = DFC2020Dataset(
            dataframe=df,
            root_dir="/ds2/remote_sensing/DFC2020",
            split="test",
            transform=train_transform,
        )
    elif dataset_name == "spacenet":
        image_dir = "/ds2/remote_sensing/spacenet/spacenet1/3band"
        geojson_dir = "/ds2/remote_sensing/spacenet/spacenet1/geojson"

        def load_indices(path):
            with open(path) as f:
                return [int(line.strip()) for line in f]

        train_ids = load_indices("/ds2/remote_sensing/spacenet/spacenet1/train.txt")
        val_ids = load_indices("/ds2/remote_sensing/spacenet/spacenet1/val.txt")

        train_transforms = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

        test_transforms = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

        train_dataset = SpaceNetBuildingDataset(
            image_dir, geojson_dir, indices=train_ids, transform=train_transforms
        )
        val_dataset = SpaceNetBuildingDataset(
            image_dir, geojson_dir, indices=val_ids, transform=test_transforms
        )

    elif dataset_name == "dior":
        train_dataset = DIORDataset(
            root="/ds2/remote_sensing/dior/DIOR-dataset",
            split="train",
            image_size=800,
            merge_train_val=False,
            hflip_prob=0,
            color_jitter=0,
            normalize=True,
        )

        val_dataset = DIORDataset(
            root="/ds2/remote_sensing/dior/DIOR-dataset",
            split="val",
            image_size=800,
            merge_train_val=False,
            hflip_prob=0,
            color_jitter=0,
            normalize=True,
        )

        print(train_dataset[0][0].shape)
        print(val_dataset[0][0].shape)
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=detection_collate,  # Subset(train_dataset, indices=range(10000))
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=detection_collate,  # Subset(val_dataset, indices=range(2000))
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader
    elif dataset_name == "sen1floods11":
        train_dataset = Sen1Floods11HandLabeledDataset(
            split="train", resize_to=(224, 224)
        )
        val_dataset = Sen1Floods11HandLabeledDataset(split="val", resize_to=(224, 224))

        print(train_dataset[0][0].shape)
        print(val_dataset[0][0].shape)
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader
    elif dataset_name == "m-eurosat":
        train_dataset = GeoBenchDataset(
            benchmark_name="classification_v1.0",
            task_name="m-eurosat",
            split="train",
            resize=(224, 224),
        )
        val_dataset = GeoBenchDataset(
            benchmark_name="classification_v1.0",
            task_name="m-eurosat",
            split="valid",
            resize=(224, 224),
        )
        print(train_dataset[0][0].shape)
        print(val_dataset[0][0].shape)
    elif dataset_name == "m-so2sat":
        train_dataset = GeoBenchDataset(
            benchmark_name="classification_v1.0",
            task_name="m-so2sat",
            split="train",
            resize=(224, 224),
        )
        val_dataset = GeoBenchDataset(
            benchmark_name="classification_v1.0",
            task_name="m-so2sat",
            split="valid",
            resize=(224, 224),
        )
        print(train_dataset[0][0].shape)
        print(val_dataset[0][0].shape)
    elif dataset_name == "m-brick-kiln ":
        train_dataset = GeoBenchDataset(
            benchmark_name="classification_v1.0",
            task_name="m-brick-kiln",
            split="train",
            resize=(224, 224),
        )
        val_dataset = GeoBenchDataset(
            benchmark_name="classification_v1.0",
            task_name="m-brick-kiln",
            split="valid",
            resize=(224, 224),
        )
        print(train_dataset[0][0].shape)
        print(val_dataset[0][0].shape)

    elif dataset_name == "m-bigearthnet":
        train_dataset = GeoBenchDataset(
            benchmark_name="classification_v1.0",
            task_name="m-bigearthnet",
            split="train",
            resize=(224, 224),
        )
        val_dataset = GeoBenchDataset(
            benchmark_name="classification_v1.0",
            task_name="m-bigearthnet",
            split="valid",
            resize=(224, 224),
        )
        print(train_dataset[0][0].shape)
        print(val_dataset[0][0].shape)
    else:
        pass

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def evaluate(model, dataloader, criterion, device, task, num_classes):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    nb_batches = 0
    if task == "segmentation":
        if num_classes == 1:
            metric_eval = JaccardIndex(
                task="multiclass", num_classes=num_classes + 1, ignore_index=255
            ).to("cuda")
        else:
            metric_eval = JaccardIndex(
                task="multiclass", num_classes=num_classes, ignore_index=255
            ).to("cuda")

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation", leave=False):
            if task == "detection":
                images, _ = move_detection_batch_to_device(images, targets, device)
            else:
                images, targets = images.to(device), targets.to(device)
            if task == "segmentation":
                # targets = targets.long()
                if num_classes < 2:
                    targets = targets.unsqueeze(1)
                    targets = targets.float()
            nb_batches += 1

            if task == "detection":
                # oss_dict = model(images, targets)
                # loss = sum(loss_dict.values())
                total_loss = 0
                total = 1
            else:
                outputs = model(images)

                # if task == "multilabel":
                #     targets = targets.float()
                # loss = criterion(outputs, targets)
                # if task == "multilabel":
                #     correct += compute_mAP(outputs, targets)
                # elif task == "segmentation":
                #     if num_classes < 2:
                #         probs = torch.sigmoid(outputs)
                #         preds = (probs > 0.5).int()
                #         # print(preds.squeeze().int()[0], targets.squeeze().int()[0])
                #         metric_eval.update(
                #             preds.squeeze().int(), targets.squeeze().int()
                #         )
                #     else:
                #         metric_eval.update(outputs.argmax(dim=1), targets)
                # else:
                #     preds = outputs.argmax(dim=1)
                #     correct += (preds == targets).sum().item()
                #
                if task == "multilabel":
                    targets = targets.float()
                    loss = criterion(outputs, targets)
                    correct += compute_mAP(outputs, targets)

                elif task == "segmentation":
                    loss = criterion(
                        outputs, targets.long() if num_classes > 1 else targets.float()
                    )
                    if num_classes < 2:
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).int()
                        metric_eval.update(
                            preds.squeeze().int(), targets.squeeze().int()
                        )
                    else:
                        metric_eval.update(outputs.argmax(dim=1), targets)

                else:
                    # classification
                    if num_classes == 2 and (
                        outputs.dim() == 1 or outputs.size(1) == 1
                    ):
                        # Single-logit binary head -> BCEWithLogitsLoss
                        logits = (
                            outputs.squeeze(1) if outputs.dim() == 2 else outputs
                        )  # [N]
                        targets_bin = targets.float()  # [N]
                        loss = criterion(logits, targets_bin)
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).long()
                        correct += (preds == targets.long()).sum().item()
                    else:
                        # Two-logit binary (N,2) or multiclass (N,K) -> CrossEntropy
                        loss = criterion(outputs, targets.long())
                        preds = outputs.argmax(dim=1)
                        correct += (preds == targets.long()).sum().item()

                total_loss += loss.item() * targets.size(0)
                total += targets.size(0)

    avg_loss = total_loss / total
    if task == "multilabel":
        accuracy = correct / nb_batches
    elif task == "detection":
        accuracy = evaluate_detection_map(
            model, dataloader, num_classes=num_classes + 1, iou_thresh=0.5
        )
        print(f"mAP@0.5: {accuracy:.4f}")
    elif task == "segmentation":
        accuracy = metric_eval.compute().item()
        print("metric eval: ", accuracy)
    else:
        accuracy = correct / total

    return avg_loss, accuracy


def move_detection_batch_to_device(images, targets, device):
    # images: List[Tensor[C,H,W]] OR Tensor[B,C,H,W]
    if isinstance(images, list):
        images = [im.to(device, non_blocking=True) for im in images]
    else:
        # if you accidentally have BCHW, split to list
        images = [im.to(device, non_blocking=True) for im in images.unbind(dim=0)]

    # targets: List[Dict[str, Tensor]]
    targ_out = []
    for t in targets:
        t_dev = {}
        for k, v in t.items():
            t_dev[k] = v.to(device) if torch.is_tensor(v) else v
        targ_out.append(t_dev)
    return images, targ_out


def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler, device, task, num_classes
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    nb_batches = 0

    if task == "segmentation":
        if num_classes == 1:
            miou = JaccardIndex(
                task="multiclass", num_classes=num_classes + 1, ignore_index=255
            ).to("cuda")
        else:
            miou = JaccardIndex(
                task="multiclass", num_classes=num_classes, ignore_index=255
            ).to("cuda")

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        if task == "detection":
            images, targets = move_detection_batch_to_device(images, targets, device)
        else:
            images, targets = images.to(device), targets.to(device)
        if task == "segmentation":
            if num_classes < 2:
                targets = targets.unsqueeze(1)
                targets = targets.float()
        optimizer.zero_grad()
        nb_batches += 1

        if task == "detection":
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        else:
            outputs = model(images)
            if task == "multilabel":
                targets = targets.float()
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if task != "detection":
            if task == "multilabel":
                correct += compute_mAP(outputs, targets)
            elif task == "segmentation":
                if num_classes < 2:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).int()
                    miou.update(preds.squeeze().int(), targets.squeeze().int())
                else:
                    miou.update(outputs.argmax(dim=1), targets)
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()

        try:
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
        except AttributeError:
            total_loss += loss.item() * len(targets)
            total += len(targets)

    save_checkpoint(
        model, optimizer, path="checkpoints/vitseg_sen1floods11_geosane.pth"
    )

    avg_loss = total_loss / total
    if task == "multilabel":
        accuracy = correct / nb_batches
    elif task == "segmentation":
        accuracy = miou.compute().item()
    elif task == "detection":
        accuracy = evaluate_detection_map(
            model, dataloader, num_classes=num_classes + 1, iou_thresh=0.5
        )
        print(f"mAP@0.5: {accuracy:.4f}")
    else:
        accuracy = correct / total
    return avg_loss, accuracy


def compute_per_class_AP(outputs, targets, class_names=None):
    if isinstance(outputs, torch.Tensor):
        outputs = torch.sigmoid(outputs).detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    num_classes = targets.shape[1]
    aps = {}

    for c in range(num_classes):
        y_true_c = targets[:, c]
        y_score_c = outputs[:, c]

        if y_true_c.sum() == 0:
            aps[class_names[c] if class_names else c] = np.nan
            continue

        ap = average_precision_score(y_true_c, y_score_c)
        aps[class_names[c] if class_names else c] = ap

    valid_aps = [v for v in aps.values() if not np.isnan(v)]
    mean_ap = np.mean(valid_aps) if valid_aps else 0.0

    return aps, mean_ap


# def compute_mAP(outputs, targets):
#     if isinstance(outputs, torch.Tensor):
#         outputs = torch.sigmoid(outputs).detach().cpu().numpy()
#     if isinstance(targets, torch.Tensor):
#         targets = targets.detach().cpu().numpy()

#     num_classes = targets.shape[1]
#     # aps = []
#     # for c in range(num_classes):
#     #     y_true_c = targets[:, c]
#     #     y_score_c = outputs[:, c]

#     #     if y_true_c.sum() == 0:
#     #         continue
#     #     print(y_true_c, y_score_c)
#     #     ap = average_precision_score(y_true_c, y_score_c)
#     #     aps.append(ap)

#     # print(aps)

#     aps = []
#     for c in range(num_classes):
#         y_true_c = targets[:, c]
#         y_score_c = outputs[:, c]

#         if y_true_c.sum() == 0:
#             aps.append(np.nan)  # preserve position
#             continue

#         ap = average_precision_score(y_true_c, y_score_c)
#         aps.append(ap)

#     # Use np.nanmean to ignore skipped classes safely
#     return np.nanmean(aps)

#     # if len(aps) == 0:
#     #     return 0.0
#     # return np.mean(aps)


def compute_mAP(outputs, targets):
    # Convert to numpy arrays
    if isinstance(outputs, torch.Tensor):
        if outputs.min() < 0 or outputs.max() > 1:  # logits
            outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    assert outputs.shape == targets.shape, (
        "outputs and targets must have the same shape"
    )

    aps = []
    for c in range(targets.shape[1]):
        y_true_c = targets[:, c]
        y_score_c = outputs[:, c]

        if y_true_c.sum() == 0:
            aps.append(np.nan)
            continue

        ap = average_precision_score(y_true_c, y_score_c)
        aps.append(ap)

    return np.nanmean(aps)
