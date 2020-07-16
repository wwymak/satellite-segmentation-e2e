import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def albumentations_transforms():
    transform = A.Compose([
        # A.ToFloat(max_value=65535.0),
        A.ToFloat(max_value=1.0),
        A.RandomSizedCrop(min_max_height=(512, 512), width=256, height=256),
        A.RandomRotate90(),
        A.Flip(),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
        ], p=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
        # A.Normalize(),
        ToTensorV2(),
    ])
    return transform

