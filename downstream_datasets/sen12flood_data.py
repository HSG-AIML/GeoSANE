import json
import os
import glob

import cv2
import numpy as np
import torch
import torch.utils.data
import rasterio as rio


S1_ALL = ["VV", "VH"]
S1_CEIL = {"VV": 10.0, "VH": 10.0}
CORRUPTED_FILES = [
    "0104/S1B_IW_GRDH_1SDV_20181223T181742_20181223T181807_014173_01A569_9A1D",
    # "S1B_IW_GRDH_1SDV_20190122T032731_20190122T032756_014601_01B34A_9A7F",
]


class SEN12FLOOD(torch.utils.data.Dataset):
    s1_all = S1_ALL
    s1_ceil = S1_CEIL

    def __init__(
        self, data_dir=None, split="train", bands=None, transforms=None, img_size=224
    ):
        self.data_dir = data_dir
        self.bands = bands
        self.split = split
        self.transforms = transforms
        self.classes = ["flooded", "not_flooded"]
        self.img_size = img_size

        # read list of train/test/val sample locations
        with open(os.path.join(self.data_dir, f"{split}_locations.txt")) as f:
            self.split_locations = [x.strip() for x in f.readlines()]

        # read all labels
        with open(os.path.join(self.data_dir, "S1list.json")) as f:
            self.label_data = json.load(f)

        # remove labels from wrong splits
        for k in list(self.label_data.keys()):
            if k not in self.split_locations:
                del self.label_data[k]

        assert len(self.split_locations) == len(
            self.label_data
        ), f"{len(self.split_locations)=}, {len(self.label_data)=}"

        # create list of sample dicts
        # self.locations = self.label_data.keys()
        self.locations = list(self.label_data.keys())
        self.samples = []
        for location_num in self.locations:
            location = self.label_data[location_num]
            for event_num in range(1, int(location["count"])):
                event_data = location[str(event_num)]
                flood_label = int(event_data["FLOODING"])
                filename = event_data["filename"]
                if location["folder"] + "/" + filename in CORRUPTED_FILES:
                    print(f"Skipping corrupted file: {filename=}")
                    continue
                filepaths = glob.glob(
                    os.path.join(self.data_dir, location["folder"], filename + "*")
                )
                if len(filepaths) != 2:
                    print(
                        f"Skipping missing file {filepaths=}, {location['folder']=}, {filename=}"
                    )
                    continue

                self.samples.append({"filepaths": filepaths, "label": flood_label})

    def __getitem__(self, idx):
        meta = self.samples[idx]

        bands = {}
        for file in meta["filepaths"]:
            with rio.open(file) as dataset:
                data = dataset.read()

                if "VV" in self.bands and file.endswith("VV.tif"):
                    data[data > self.s1_ceil["VV"]] = self.s1_ceil["VV"]
                    bands["VV"] = data
                elif "VH" in self.bands and file.endswith("VH.tif"):
                    data[data > self.s1_ceil["VH"]] = self.s1_ceil["VH"]
                    bands["VH"] = data
                else:
                    raise ValueError(f"File not identified as VV or VH: {file}")

        if len(bands) != 2:
            print(bands.keys(), meta["filepaths"])

        # make sure band order is correct
        data = []
        for band in self.bands:
            data.append(bands[band])

        # stack bands
        data = np.moveaxis(np.concatenate(data), 0, -1)

        # make sure all images have the same shape, to stack in batch
        data = cv2.resize(
            data,
            dsize=(self.img_size, self.img_size),
            interpolation=cv2.INTER_CUBIC,
        )

        sample = {}
        sample["label"] = torch.tensor(meta["label"], dtype=torch.long)
        sample["image"] = torch.tensor(
            np.moveaxis(data, -1, 0), dtype=torch.float32
        )  # c,h,w

#         if self.transforms:
#             sample = self.transforms(sample)

        return sample["image"], sample["label"]

    def __len__(self):
        return len(self.samples)
