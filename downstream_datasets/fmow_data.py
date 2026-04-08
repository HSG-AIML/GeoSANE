import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class fMoWDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory where the "training" and "validation" folders are located.
            split (str): "train" or "val", depending on the dataset split you are using.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Set the directory based on the split (training or validation)
        self.data_dir = os.path.join(root_dir, split)

        # Initialize a list to store image paths and labels
        self.image_paths = []
        self.labels = []

        # Iterate over each class (i.e., subfolder like "airport", "amusement_park", etc.)
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)

            if os.path.isdir(class_path):
                # Traverse each subfolder (e.g., "airport_0", "airport_108", etc.)
                for subfolder in os.listdir(class_path):
                    subfolder_path = os.path.join(class_path, subfolder)

                    # If the subfolder contains files, look for the '0_rgb.jpg' files
                    if os.path.isdir(subfolder_path):
                        for file_name in os.listdir(subfolder_path):
                            # Check if the file is the one we want ('0_rgb.jpg')
                            if file_name.endswith('0_rgb.jpg'):
                                # Add the file path and class label to the lists
                                img_path = os.path.join(subfolder_path, file_name)
                                self.image_paths.append(img_path)
                                self.labels.append(class_name)

        # Class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(self.data_dir)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image path and label
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open the image
        image = Image.open(img_path).convert("RGB")

        # Apply the transformation (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        # Convert the class label to its corresponding index
        label_idx = self.class_to_idx[label]

        return image, label_idx
