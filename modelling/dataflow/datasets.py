from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
from PIL import Image
from fastcore.utils import *

import numpy as np

class SatelliteSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_id_list, transform=None,preprocessing=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_id_list = image_id_list
        self.transform = transform
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        image_id = self.image_id_list[index]
        image_filepath = self.image_dir / f"RGB-PanSharpen_{image_id}.tif"
        mask_filepath = self.mask_dir / f"mask_{image_id}.png"
        img = io.imread(image_filepath, plugin='tifffile')
        # rescale tifffile
        img = (img - img.min())/(img.max() - img.min()).copy().astype(np.float32)
        mask = io.imread(mask_filepath).copy().astype(np.float32)
        # mask = np.asarray(Image.open(mask_filepath))

        # apply preprocessing for smp
        if self.preprocessing:
            preprocesed = self.preprocessing(image=img, mask=mask)
            img, mask = preprocesed['image'], preprocesed['mask']

        if self.transform:
            augmentation = self.transform(image=img, mask=mask)
            img = augmentation['image']
            mask = augmentation['mask']



        sample = {"image": img, "mask": mask}
        return sample

    def __len__(self):
        return len(self.image_id_list)
