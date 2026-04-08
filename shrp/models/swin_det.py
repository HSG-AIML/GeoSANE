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

class SwinBackbone(nn.Module):
    def __init__(self, model_name="swin_s3_base_224", pretrained=False, in_chans=3, img_size=800):
        super().__init__()
        swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            img_size=(img_size, img_size),
        )
        swin.head = nn.Identity()  # drop classifier head
        self.swin = swin
        self.out_channels = swin.num_features  # 1024 for base

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)

        feat = self.swin.forward_features(x)  # can be [B,H,W,C] or [B,C,H,W]

        if feat.ndim == 4 and feat.shape[1] != self.out_channels:
            # timm sometimes returns channels-last → permute
            feat = feat.permute(0, 3, 1, 2).contiguous()

        return {"0": feat}




class SwinDet(nn.Module):
    """
    Faster R-CNN with Swin backbone.
    """
    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        num_classes: int = 91,
        num_channels: int = 3,
        anchor_sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
        pretrained_backbone: bool = False,
        img_size=800,
    ):
        super().__init__()
        backbone = SwinBackbone(model_name, pretrained=pretrained_backbone, in_chans=num_channels, img_size=800)
        rpn_anchors = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchors,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            min_size=800,
            max_size=800,
        )

    @property
    def backbone(self) -> SwinBackbone:
        return self.detector.backbone

    def forward(self, images, targets: Optional[list] = None):
        return self.detector(images, targets)

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

def load_classifier_ckpt_into_swinnet(det_model: SwinDet, ckpt: Dict[str, Any] | str) -> None:
    """
    Map a Swin classification checkpoint into the detection backbone.
    Drops classifier head (head.fc.weight/bias).
    """
    if isinstance(ckpt, str):
        raw = torch.load(ckpt, map_location="cpu")
    else:
        raw = ckpt
    sd = _unwrap_state_dict(raw)
    sd = _strip_prefixes(sd)

    # drop classifier head if present
    for bad in ("head.weight", "head.bias", "head.fc.weight", "head.fc.bias", "fc.weight", "fc.bias"):
        sd.pop(bad, None)

    keep: Dict[str, torch.Tensor] = {}
    bb = det_model.backbone.swin
    bb_keys = set(name for name, _ in bb.named_parameters()) | set(name for name, _ in bb.named_buffers())
    for k, v in sd.items():
        if k in bb_keys:
            keep[k] = v

    missing, unexpected = bb.load_state_dict(keep, strict=False)
    print("[load_classifier_ckpt_into_swinnet] loaded backbone weights")
    if missing:
        print(" - missing keys:", len(missing))
        print("   ", "\n    ".join(missing[:20]), "..." if len(missing) > 20 else "")
    if unexpected:
        print(" - unexpected keys:", len(unexpected))
        print("   ", "\n    ".join(unexpected[:20]), "..." if len(unexpected) > 20 else "")
