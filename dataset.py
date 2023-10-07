import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        '''
            - The input image is already in RGB, just to be on safer side, the image is converted to RGB.
            - The mask is a grayscale image.
            - The image and mask objects are converted to numpy array for albumentation process (data augmentation)

        '''
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        '''
        A simple normalization. -> For black it is 0.0 and white it is 255.0 (Just converting the 255.0 to 1.0)
        '''
        mask[mask==255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask