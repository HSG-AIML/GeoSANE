import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np

import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np

class BigEarthNetS2Dataset(Dataset):
    def __init__(self, df, root_dir, split="train", label_map=None, transform=None, return_meta=False):
        """
        Args:
            df (pd.DataFrame): DataFrame with 'patch_id', 'labels', and 'split'
            root_dir (str): Root path to BigEarthNet-S2 directory
            split (str): 'train', 'val', or 'test'
            label_map (dict): maps label strings to indices
            transform (callable): optional transform applied to image
            return_meta (bool): if True, returns patch_id as well
        """
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.return_meta = return_meta

        self.band_list = [
            "B01", "B02", "B03", "B04", "B05", "B06",
            "B07", "B08", "B8A", "B09", "B11", "B12"
        ]

        self.label_map = label_map or self._build_label_map()

    def _build_label_map(self):
        all_labels = set(label for labels in self.df["labels"] for label in labels)
        return {label: i for i, label in enumerate(sorted(all_labels))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_id = row["patch_id"]
        label_names = row["labels"]

        # Extract tile directory name (first 7 parts of patch_id joined by '_')
        tile_dir = "_".join(patch_id.split("_")[:6])
        patch_dir = os.path.join(self.root_dir, tile_dir, patch_id)

        target_size = (224, 224)
        bands = []
        for band in self.band_list:
            band_path = os.path.join(patch_dir, f"{patch_id}_{band}.tif")
            with rasterio.open(band_path) as src:
                band_array = src.read(
                    1,
                    out_shape=target_size,
                    resampling=rasterio.enums.Resampling.bilinear
                )
            bands.append(band_array)

        image = np.stack(bands, axis=0).astype(np.float32)  # [12, 120, 120]
                # Optional transform
        if self.transform:
            # Albumentations expects HWC, not CHW
            image = image.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)

            augmented = self.transform(image=image)
            image = augmented["image"]  # Back to tensor (C, H, W)

        image = image / 10000
        # Multi-hot label vector
        label = torch.zeros(len(self.label_map), dtype=torch.float32)
        for lbl in label_names:
            if lbl in self.label_map:
                label[self.label_map[lbl]] = 1.0

        if self.return_meta:
            return image, label, patch_id
        else:
            return image, label
