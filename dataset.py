import os
import torch
import numpy as np
import tifffile as tiff
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import albumentations as A


class SatelliteDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.tif', '.png'))

        # Load image and mask
        image = tiff.imread(img_path)
        image = image.astype(np.float32)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

        if len(mask.shape) == 4 and mask.shape[1] == 1:  # Check if there's an extra singleton dimension
            mask = torch.squeeze(mask, dim=1)  # Squeeze out the unnecessary dimension

        # Apply transformations using named arguments
        if self.transform:
            augmented = self.transform(image=image)  # Named argument for image
            image = augmented['image']
        if self.target_transform:
            augmented_mask = self.target_transform(image=mask)  # Named argument for mask
            mask = augmented_mask['image']




        return image, mask
