from __future__ import annotations
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image



DIOR_CLASSES: List[str] = [
    "airplane","airport","baseballfield","basketballcourt","bridge","chimney",
    "dam","Expressway-Service-area","Expressway-toll-station","golffield",
    "groundtrackfield","harbor","overpass","ship","stadium","storagetank",
    "tenniscourt","trainstation","vehicle","windmill"
]
CLASS_TO_IDX: Dict[str, int] = {c: i + 1 for i, c in enumerate(DIOR_CLASSES)}


def _read_split_ids(root: Path, split: str, merge_train_val: bool) -> List[str]:
    """
    Read image IDs from ImageSets/Main/{split}.txt; if split=='train' and merge_train_val=True,
    concatenate train + val.
    """
    sets = [split]
    if split == "train" and merge_train_val:
        sets = ["train", "val"]

    ids: List[str] = []
    for s in sets:
        p = root / "ImageSets" / "Main" / f"{s}.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
        with p.open("r") as f:
            ids += [line.strip() for line in f if line.strip()]
    return ids


def _find_image(img_dir: Path, stem: str) -> Path:
    """
    DIOR images can be .jpg or .png depending on release/mirror.
    """
    for ext in (".jpg", ".JPG", ".jpeg", ".png"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image for id '{stem}' not found under {img_dir}")


def _load_voc_xml(xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (boxes[N,4], labels[N]) in (xmin, ymin, xmax, ymax), 1-indexed labels (background=0 unused).
    Filters out objects not in DIOR_CLASSES.
    """
    root = ET.parse(xml_path).getroot()
    boxes: List[List[float]] = []
    labels: List[int] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None:
            continue
        name = name_el.text
        if name not in CLASS_TO_IDX:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        # guard bad boxes
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[name])
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def _letterbox_resize(
    img: Image.Image,
    boxes: np.ndarray,
    target: int,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, np.ndarray]:
    """
    Resize shortest side to `target` while preserving aspect ratio, then pad to square (target x target).
    Boxes are scaled accordingly. Returns new image, scaled boxes.
    """
    w, h = img.size
    scale = target / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # create padded canvas
    canvas = Image.new("RGB", (target, target), pad_color)
    canvas.paste(img_resized, (0, 0))

    if boxes.size == 0:
        return canvas, boxes

    boxes_scaled = boxes.copy()
    boxes_scaled[:, [0, 2]] *= (new_w / w)
    boxes_scaled[:, [1, 3]] *= (new_h / h)
    return canvas, boxes_scaled


class DIORDataset(Dataset):
    """
    DIOR (horizontal boxes) dataset for PyTorch detection models.
    Directory layout:
      DIOR/
        JPEGImages/
        Annotations/
        ImageSets/Main/{train,val,test}.txt   # file contains image ids (no extension)

    Returns (image_tensor, target_dict) where target has keys:
      boxes [N,4], labels [N], image_id [1], area [N], iscrowd [N]
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 800,
        merge_train_val: bool = True,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.10,
        normalize: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.img_dir = self.root / "JPEGImages"
        self.ann_dir = self.root / "Annotations"
        self.ids = _read_split_ids(self.root, split, merge_train_val)
        self.image_size = int(image_size)
        self.hflip_prob = float(hflip_prob)
        self.color_jitter = float(color_jitter)
        self.normalize = normalize

        # sanity checks
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Missing JPEGImages dir: {self.img_dir}")
        if not self.ann_dir.exists():
            raise FileNotFoundError(f"Missing Annotations dir: {self.ann_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_path = _find_image(self.img_dir, img_id)
        xml_path = self.ann_dir / f"{img_id}.xml"

        # load
        img = Image.open(img_path).convert("RGB")
        boxes_np, labels_np = _load_voc_xml(xml_path)

        # light multiplicative brightness jitter
        if self.color_jitter > 0 and random.random() < 0.80:
            factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            img = TF.adjust_brightness(img, factor)

        # letterbox resize to square canvas (e.g., 800×800)
        img, boxes_np = _letterbox_resize(img, boxes_np, self.image_size)

        # optional horizontal flip
        if random.random() < self.hflip_prob:
            img = TF.hflip(img)
            if boxes_np.size > 0:
                w = img.width
                x1 = boxes_np[:, 0].copy()
                x2 = boxes_np[:, 2].copy()
                boxes_np[:, 0] = w - x2
                boxes_np[:, 2] = w - x1

        # to tensor
        img_t = TF.to_tensor(img)  # [0,1]
        if self.normalize:
            img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        if boxes_np.size == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.from_numpy(boxes_np).float()
            labels = torch.from_numpy(labels_np).long()

        # build target dict per torchvision detection API
        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        if boxes.numel() > 0:
            wh = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            hh = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            target["area"] = (wh * hh).to(torch.float32)
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        return img_t, target


def dior_collate_fn(batch):
    """
    Collate function for detection: returns List[Tensor], List[Dict]
    """
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)





ds = DIORDataset(
    root="/ds2/remote_sensing/dior/DIOR-dataset",
    split="train",
    image_size=224,
    merge_train_val=False,
    hflip_prob=0,
    color_jitter=0,
    normalize=True,
)
