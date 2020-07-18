import random
import torch
import cv2
from matplotlib import pyplot as plt

import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

def get_train_augmentation():
    transform = albu.Compose([
        # A.ToFloat(max_value=65535.0),
        albu.ToFloat(max_value=1.0),
        albu.RandomSizedCrop(min_max_height=(512, 512), width=256, height=256),
        albu.Rotate(),
        albu.Flip(),
        albu.OneOf([
            albu.MotionBlur(p=0.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        albu.OneOf([
            albu.OpticalDistortion(p=0.3),
            albu.GridDistortion(p=0.1),
        ], p=0.2),
        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
        # A.Normalize(),
        ToTensorV2(),
    ])
    return transform

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.ToFloat(max_value=1.0),
        albu.RandomSizedCrop(min_max_height=(512, 512), width=256, height=256),
        ToTensorV2(),
    ]
    return albu.Compose(test_transform)


def denormalize(t, mean, std, max_pixel_value=255):
    assert isinstance(t, torch.Tensor), "{}".format(type(t))
    assert t.ndim == 3
    d = t.device
    mean = torch.tensor(mean, device=d).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std, device=d).unsqueeze(-1).unsqueeze(-1)
    tensor = std * t + mean
    tensor *= max_pixel_value
    return tensor


def prepare_batch_fp32(batch, device, non_blocking):
    x, y = batch["image"], batch["mask"]
    x = convert_tensor(x, device, non_blocking=non_blocking)
    y = convert_tensor(y, device, non_blocking=non_blocking).long()
    return x, y

