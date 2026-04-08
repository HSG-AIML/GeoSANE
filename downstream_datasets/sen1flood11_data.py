import os
import csv
from typing import Optional, Callable, Dict, Any, Tuple, List

import torch
from torch.utils.data import Dataset
import rasterio
import torch.nn.functional as F


class Sen1Floods11HandLabeledDataset(Dataset):
    """
    Sen1Floods11 hand-labeled flood events (S1Hand + LabelHand).

    Returns:
      {
        "image": FloatTensor [2,H,W],   # Sentinel-1 VV/VH
        "label": LongTensor  [H,W],     # {0,1}, where -1 is remapped to 0
        "id": str
      }
    """

    DEFAULT_MEAN = (0.6851, 0.5235)
    DEFAULT_STD  = (0.0820, 0.1102)

    def __init__(
        self,
        data_root: str = "/ds2/remote_sensing/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand",
        label_root: str = "/ds2/remote_sensing/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand",
        splits_dir: str = "/ds2/remote_sensing/sen1floods11/v1.1/splits/flood_handlabeled",
        split: str = "train",                        # "train" or "val"
        transform: Optional[Callable] = None,
        normalize: bool = True,
        mean: Optional[Tuple[float, float]] = None,
        std: Optional[Tuple[float, float]] = None,
        resize_to: Optional[Tuple[int, int]] = None  # e.g. (224,224)
    ):
        assert split in {"train", "val"}
        self.data_root = data_root
        self.label_root = label_root
        self.transform = transform
        self.normalize = normalize
        self.mean = mean if mean is not None else self.DEFAULT_MEAN
        self.std = std if std is not None else self.DEFAULT_STD
        self.resize_to = resize_to

        split_csv = os.path.join(
            splits_dir,
            "flood_train_data.csv" if split == "train" else "flood_test_data.csv"
        )
        if not os.path.isfile(split_csv):
            raise FileNotFoundError(f"Split file not found: {split_csv}")

        self.samples: List[Tuple[str, str]] = []
        with open(split_csv, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                img_fn, label_fn = row[0].strip(), row[1].strip()
                img_path = os.path.join(self.data_root, img_fn)
                label_path = os.path.join(self.label_root, label_fn)
                self.samples.append((img_path, label_path))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_tif(path: str) -> torch.Tensor:
        with rasterio.open(path) as src:
            arr = src.read()  # [C,H,W]
        return torch.from_numpy(arr.astype("float32"))

    def _zscore(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)[:, None, None]
        return (x - mean) / std

    @staticmethod
    def _extract_id(img_filename: str) -> str:
        base = os.path.basename(img_filename)
        if base.endswith("_S1Hand.tif"):
            return base.replace("_S1Hand.tif", "")
        return os.path.splitext(base)[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label_path = self.samples[idx]

        img = self._read_tif(img_path)      # [2,H,W]
        lab = self._read_tif(label_path)    # [1,H,W]
        label = lab.squeeze(0).to(torch.long)

        # remap -1 → 0
        label[label == -1] = 0

        img = img.to(torch.float32)
        if self.normalize:
            img = self._zscore(img)

        # resize if needed
        if self.resize_to is not None:
            h, w = self.resize_to
            img = F.interpolate(img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
            label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(),
                                  size=(h, w),
                                  mode="nearest").squeeze(0).squeeze(0).to(torch.long)

        if self.transform is not None:
            transformed = self.transform(image=img, label=label)
            img = transformed["image"]
            label = transformed["label"]

        img[torch.isnan(img)] = 0

        return img, label
