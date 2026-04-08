# models/vit_det.py
from __future__ import annotations
from collections import OrderedDict
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from timm.layers import to_2tuple

def modify_vit_input_channels(vit: nn.Module, in_chans: int) -> None:
    """
    Modify ViT patch_embed.proj to accept `in_chans` inputs.
    Works for timm ViT (e.g., vit_large_patch16_224).
    If in_chans != old_in, we (best-effort) copy/avg/replicate weights.
    """
    if not hasattr(vit, "patch_embed") or not hasattr(vit.patch_embed, "proj"):
        raise AttributeError("Unexpected ViT structure: missing patch_embed.proj")

    old = vit.patch_embed.proj  # Conv2d(in_chans, embed_dim, kernel_size=patch, stride=patch)
    if in_chans == old.in_channels:
        return  # nothing to do

    new = nn.Conv2d(
        in_chans,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )

    with torch.no_grad():
        if in_chans < old.in_channels:
            # copy first in_chans
            new.weight[:, :in_chans].copy_(old.weight[:, :in_chans])
            if new.weight.shape[1] > in_chans:
                # zero the rest
                new.weight[:, in_chans:].zero_()
        else:
            # in_chans > old: average existing channels and replicate
            mean_w = old.weight.mean(dim=1, keepdim=True)  # [out,1,kh,kw]
            new.weight.copy_(mean_w.repeat(1, in_chans, 1, 1))
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)

    vit.patch_embed.proj = new


def _unwrap_state_dict(sd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Handle common wrappers: {'state_dict':...}, {'model':...}, etc."""
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "net", "module"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd


def _strip_prefixes(sd: Dict[str, torch.Tensor], prefixes=("module.", "model.", "backbone.", "vit.")) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        out[k] = v
    return out



class ViTBackbone(nn.Module):
    """
    Wrap a timm ViT (e.g., vit_large_patch16_224) so it returns a single spatial feature map:
      forward(x) -> {"0": [B, C, H/16, W/16]}
    `out_channels` is set to C (embed_dim), as required by torchvision detection heads.
    """
    def __init__(self, model_name: str = "vit_large_patch16_224", pretrained: bool = False, in_chans: int = 3):
        super().__init__()
        vit = timm.create_model(model_name, pretrained=pretrained, features_only=False, in_chans=3)
        vit.patch_embed.img_size = (800, 800)
        if in_chans != 3:
            modify_vit_input_channels(vit, in_chans)

        self.patch_embed = vit.patch_embed
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed        # [1, 1+Gh*Gw, C] absolute pos embed
        self.pos_drop = vit.pos_drop
        self.embed_dim = vit.embed_dim

        self.patch_size: Tuple[int, int] = to_2tuple(getattr(self.patch_embed, "patch_size", 16))
        grid = getattr(self.patch_embed, "grid_size", (14, 14))
        self.grid_h, self.grid_w = grid[0], grid[1]

        self.out_channels = self.embed_dim     # required by torchvision detection

    @torch.no_grad()
    def _interp_pos_embed(self, nH: int, nW: int, device: torch.device) -> torch.Tensor:
        # split CLS vs patch tokens
        pos = self.pos_embed.to(device)  # [1, 1+Gh*Gw, C]
        cls_pos, patch_pos = pos[:, :1, :], pos[:, 1:, :]
        Gh, Gw = self.grid_h, self.grid_w
        patch_pos = patch_pos.reshape(1, Gh, Gw, self.embed_dim).permute(0, 3, 1, 2)  # [1,C,Gh,Gw]
        patch_pos = F.interpolate(patch_pos, size=(nH, nW), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, nH * nW, self.embed_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)  # [1, 1+nH*nW, C]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, in_chans, H, W]
        returns: {"0": feat} where feat is [B, C, H/16, W/16]
        """
        B, _, H, W = x.shape

        # (1) patchify -> [B, N, C], N = (H/ps)*(W/ps)
        x = self.patch_embed(x)
        nH, nW = H // self.patch_size[0], W // self.patch_size[1]

        # (2) prepend CLS, add/interp pos embed
        cls = self.cls_token.expand(B, -1, -1)                 # [B,1,C]
        x = torch.cat([cls, x], dim=1)                         # [B,1+N,C]
        x = x + self._interp_pos_embed(nH, nW, x.device)       # match spatial grid
        x = self.pos_drop(x)

        # (3) transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # (4) drop CLS, reshape tokens back to map
        x = x[:, 1:, :]                                        # [B,N,C]
        feat = x.transpose(1, 2).reshape(B, self.embed_dim, nH, nW)  # [B,C,nH,nW]
        return {"0": feat}


class ViTDet(nn.Module):
    """
    End-to-end detection model:
      - ViT backbone returning a spatial feature map
      - Faster R-CNN head (single-level, stride=16)
    forward(images, targets?) -> same API as torchvision FasterRCNN
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        num_channels: int = 3,
        anchor_sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        backbone = ViTBackbone(model_name=model_name, pretrained=pretrained_backbone, in_chans=num_channels)
        rpn_anchors = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,            # includes background
            rpn_anchor_generator=rpn_anchors,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            min_size=800,
            max_size=800
        )

    @property
    def backbone(self) -> ViTBackbone:
        return self.detector.backbone  # convenience accessor

    def forward(self, images, targets: Optional[list] = None):
        return self.detector(images, targets)

def load_classifier_ckpt_into_vitdet(det_model: ViTDet, ckpt: Dict[str, Any] | str) -> None:
    """
    Map a ViT-L classification checkpoint into the wrapped backbone:
      - drops classification head (e.g., head.weight/bias)
      - tolerates common wrappers/prefixes
    Use BEFORE training detection.
    """
    if isinstance(ckpt, str):
        raw = torch.load(ckpt, map_location="cpu")
    else:
        raw = ckpt
    sd = _unwrap_state_dict(raw)
    sd = _strip_prefixes(sd)

    # remove classifier head if present
    for bad in ("head.weight", "head.bias", "fc.weight", "fc.bias", "classifier.weight", "classifier.bias"):
        sd.pop(bad, None)

    # We only load keys that exist in our backbone's named buffers/params
    keep: Dict[str, torch.Tensor] = {}
    bb = det_model.backbone
    bb_keys = set(name for name, _ in bb.named_parameters()) | set(name for name, _ in bb.named_buffers())

    for k, v in sd.items():
        if k in bb_keys:
            keep[k] = v

    missing, unexpected = bb.load_state_dict(keep, strict=False)
    print("[load_classifier_ckpt_into_vitdet] loaded backbone weights")
    if missing:
        print(" - missing keys:", len(missing))
        print("   ", "\n    ".join(missing[:20]), "..." if len(missing) > 20 else "")
    if unexpected:
        print(" - unexpected keys:", len(unexpected))
        print("   ", "\n    ".join(unexpected[:20]), "..." if len(unexpected) > 20 else "")
