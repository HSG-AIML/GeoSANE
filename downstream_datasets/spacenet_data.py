import os
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
import geopandas as gpd
import numpy as np

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

class SpaceNetBuildingDataset(Dataset):
    def __init__(self, image_dir, geojson_dir, indices=None, transform=None):
        self.image_dir = image_dir
        self.geojson_dir = geojson_dir
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith('.tif')
        ])
        if indices is not None:
            self.image_files = [self.image_files[i] for i in indices]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image_id = image_filename.split('_')[-1].split('.')[0]  # e.g., "img4551"


        image_path = os.path.join(self.image_dir, image_filename)
        geojson_filename = f"Geo_{'_'.join(image_filename.split('_')[1:])}".replace('.tif', '.geojson')
        geojson_path = os.path.join(self.geojson_dir, geojson_filename)

        with rasterio.open(image_path) as src:
            image = src.read()  # shape: (bands, H, W)
            # image = np.clip(image, 0, 10000) / 10000.0  # Normalize to [0,1]
            height, width = src.height, src.width
            bounds = src.bounds
            transform = src.transform


        if os.path.exists(geojson_path):
            geo_df = gpd.read_file(geojson_path).to_crs("EPSG:3857")  # ensure projected CRS
            shapes = geo_df['geometry'].values

            # Get bounds of all polygons
            minx, miny, maxx, maxy = geo_df.total_bounds


            # Define transform to map bounds to mask grid
            transform = from_bounds(minx, miny, maxx, maxy, width, height)

            mask = geometry_mask(
                geometries=shapes,
                transform=transform,
                out_shape=(height, width),
                invert=True
            ).astype(np.uint8)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        image = image.transpose(1, 2, 0)

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']


        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
