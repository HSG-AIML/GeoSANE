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

import torchvision

class ResNetBackbone(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=False, in_chans=3):
        super().__init__()
        resnet = torchvision.models.__dict__[model_name](pretrained=pretrained)

        # fix input channels if needed
        if in_chans != 3:
            conv1 = resnet.conv1
            new_conv = nn.Conv2d(
                in_chans, conv1.out_channels, kernel_size=conv1.kernel_size,
                stride=conv1.stride, padding=conv1.padding, bias=conv1.bias is not None,
            )
            with torch.no_grad():
                if in_chans < 3:
                    new_conv.weight[:, :in_chans] = conv1.weight[:, :in_chans]
                    new_conv.weight[:, in_chans:] = 0
                else:
                    mean_w = conv1.weight.mean(dim=1, keepdim=True)
                    new_conv.weight = mean_w.repeat(1, in_chans, 1, 1)
            resnet.conv1 = new_conv

        # keep named layers instead of nn.Sequential
        self.conv1 = resnet.conv1
        self.bn1   = resnet.bn1
        self.relu  = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.out_channels = 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return {"0": x}



class ResNetDet(nn.Module):
    """
    End-to-end detection model:
      - ResNet backbone
      - Faster R-CNN head
    """
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 91,
        num_channels: int = 3,
        anchor_sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        backbone = ResNetBackbone(model_name=model_name, pretrained=pretrained_backbone, in_chans=num_channels)
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
    def backbone(self) -> ResNetBackbone:
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


def load_classifier_ckpt_into_resnetdet(det_model: ResNetDet, ckpt: Dict[str, Any] | str) -> None:
    """
    Map a ResNet classification checkpoint into the wrapped backbone:
      - drops classifier head (fc.weight/bias)
      - tolerates common wrappers/prefixes
    """
    if isinstance(ckpt, str):
        raw = torch.load(ckpt, map_location="cpu")
    else:
        raw = ckpt
    sd = _unwrap_state_dict(raw)
    sd = _strip_prefixes(sd)

    # remove classifier head if present
    for bad in ("fc.weight", "fc.bias", "classifier.weight", "classifier.bias"):
        sd.pop(bad, None)

    # Only keep keys matching backbone
    keep: Dict[str, torch.Tensor] = {}
    bb = det_model.backbone
    bb_keys = set(name for name, _ in bb.named_parameters()) | set(name for name, _ in bb.named_buffers())
    for k, v in sd.items():
        if k in bb_keys:
            keep[k] = v

    missing, unexpected = bb.load_state_dict(keep, strict=False)
    print("[load_classifier_ckpt_into_resnetdet] loaded backbone weights")
    if missing:
        print(" - missing keys:", len(missing))
        print("   ", "\n    ".join(missing[:20]), "..." if len(missing) > 20 else "")
    if unexpected:
        print(" - unexpected keys:", len(unexpected))
        print("   ", "\n    ".join(unexpected[:20]), "..." if len(unexpected) > 20 else "")
