import json
import os
import glob

import cv2
import numpy as np
import torch
import torch.utils.data
# import rasterio as rio

from sklearn.model_selection import train_test_split
from PIL import Image


S2_SWIR_RE4 = ["B11", "B12", "B8A"]


class CaliforniaWildfire(torch.utils.data.Dataset):
    s2_swir_re4 = S2_SWIR_RE4

    def __init__(
        self, data_dir=None, split="train", bands=None, transforms=None, img_size=224
    ):
        self.data_dir = data_dir
        self.bands = S2_SWIR_RE4
        self.transforms = transforms
        self.classes = ["fire", "no-fire"]
        self.img_size = img_size
        self.split = split

        self.ids, self.labels = self.get_split_ids(split=self.split)

    def get_split_ids(self, split):
        fires = os.listdir(os.path.join(self.data_dir, 'fire'))
        no_fires = os.listdir(os.path.join(self.data_dir, 'no-fire'))

        fire_ids = np.unique([int(i.split('_')[0]) for i in fires])
        no_fire_ids = np.unique([int(i.split('_')[0]) for i in no_fires])

        train_fire, test_fire = train_test_split(fire_ids, test_size=0.33, random_state=42)
        train_no_fire, test_no_fire = train_test_split(no_fire_ids, test_size=0.33, random_state=42)

        if split == 'train':
            images = list(train_fire) + list(train_no_fire)
            labels = [1] * len(train_fire) + [0] * len(train_no_fire)
        else:
            images = list(test_fire) + list(test_no_fire)
            labels = [1] * len(test_fire) + [0] * len(test_no_fire)
        return images, labels


    def __getitem__(self, idx):
        image_id = self.ids[idx]
        label = self.labels[idx]

        bands = {}
        if label == 1:
            data_path = os.path.join(self.data_dir, 'fire')
        else:
            data_path = os.path.join(self.data_dir, 'no-fire')

        data = []
        for b in self.bands:
            img_band = str(image_id) + '_' + b + '.tif'
            # with rio.open(os.path.join(data_path, img_band)) as f:
            #     img_data_band = f.read()
            img_data_band = np.array(Image.open(os.path.join(data_path, img_band)))

            data.append(img_data_band)

        # stack bands
        data = np.moveaxis(np.stack(data, axis=0), 0, -1)
        # print(data.shape)
        # make sure all images have the same shape, to stack in batch
        data = cv2.resize(
            data,
            dsize=(self.img_size, self.img_size),
            interpolation=cv2.INTER_CUBIC,
        )

        sample = {}
        sample["label"] = torch.tensor(label, dtype=torch.long)
        sample["image"] = torch.tensor(
            np.moveaxis(data, -1, 0), dtype=torch.float32
        )  # c,h,w

        if self.transforms:
            sample = self.transforms(sample)

        return sample["image"], sample["label"]

    def __len__(self):
        return len(self.ids)
