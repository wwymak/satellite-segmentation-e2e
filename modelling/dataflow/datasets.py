from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
from PIL import Image
from fastcore.utils import *

import numpy as np

class SatelliteSegmentationDataset(Dataset):
    def __init__(self, image_filepath_list, mask_filepath_list, transform=None,preprocessing=None):
        super().__init__()
        self.image_filepath_list = image_filepath_list
        self.mask_filepath_list = mask_filepath_list
        self.transform = transform
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        image_filepath = self.image_filepath_list[index]
        mask_filepath = self.mask_filepath_list[index]
        img = io.imread(image_filepath, plugin='tifffile')
        # rescale tifffile
        if img.max() > img.min():
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
        return len(self.image_filepath_list)
