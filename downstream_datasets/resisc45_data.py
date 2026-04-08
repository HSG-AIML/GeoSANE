import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class RESISC45Dataset(Dataset):
    def __init__(self, data_dir, split_file, transform=None):
        """
        Args:
            data_dir (str): Directory with all the images organized by classes (each folder is a class).
            split_file (str): Path to the split file (e.g., train.txt, val.txt, or test.txt).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Read the split file and collect the image filenames
        with open(split_file, 'r') as f:
            self.image_filenames = [line.strip() for line in f.readlines()]

        # List all subfolders (class labels)
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Initialize a list to store image paths and their corresponding labels
        self.image_paths = []
        self.labels = []

        # Map image filenames to their full paths and labels
        for filename in self.image_filenames:
            # Extract the class name (everything before the last underscore)
            class_name = "_".join(filename.split('_')[:-1])  # Get all parts before the last underscore
            class_dir = os.path.join(data_dir, class_name)
            img_path = os.path.join(class_dir, filename)
            label = self.class_to_idx[class_name]

            self.image_paths.append(img_path)
            self.labels.append(label)

    def __len__(self):
        # Return the total number of images
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image path and label
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open the image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label
