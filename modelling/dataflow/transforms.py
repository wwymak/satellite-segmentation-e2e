import random
import torch
import cv2
from matplotlib import pyplot as plt

import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import BasicTransform
from albumentations.pytorch.functional import mask_to_tensor

class ToTensorCustom(ToTensorV2):
    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        return mask_to_tensor(mask, num_classes=params.get('num_classes', 1), sigmoid=params.get('sigmoid'))

def get_train_augmentation(image_size=256, is_display=False, normalisation_fn=None):
    # max_image_size = min(max_image_size, image_size * 2)
    # transform = albu.Compose([
    #     # A.ToFloat(max_value=65535.0),
    #     albu.ToFloat(max_value=1.0),
    #     albu.RandomSizedCrop(min_max_height=(max_image_size, max_image_size), width=image_size, height=image_size),
    #     albu.Rotate(),
    #     albu.Flip(),
    #     albu.OneOf([
    #         albu.MotionBlur(p=0.2),
    #         albu.MedianBlur(blur_limit=3, p=0.1),
    #         albu.Blur(blur_limit=3, p=0.1),
    #     ], p=0.2),
    #     albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    #     albu.OneOf([
    #         albu.OpticalDistortion(p=0.3),
    #         albu.GridDistortion(p=0.1),
    #     ], p=0.2),
    #     albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
    #     # albu.Normalize(max_pixel_value=1.),
    #     ToTensorV2(),
    # ])
    transform_list = [
        resize_transforms(image_size),
        hard_transforms(),
        post_transforms(normalisation_fn)
    ]
    if is_display:
        transform_list = transform_list[:-1]
    transform = compose(transform_list)
    return transform

def get_validation_augmentation(image_size=256, is_display=False, normalisation_fn=None):
    """Add paddings to make image shape divisible by 32"""
    # test_transform = [
    #     albu.ToFloat(max_value=1.0),
    #     albu.RandomSizedCrop(min_max_height=(image_size, image_size), width=image_size, height=image_size),
    #     # albu.Normalize(max_pixel_value=1.),
    #     ToTensorV2(),
    # ]
    transform_list = [
        pre_transforms(image_size), post_transforms(normalisation_fn)
    ]
    if is_display:
        transform_list = transform_list[:-1]
    return compose(transform_list)



def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=ToTensorV2, mask=ToTensorV2),
    ]
    return albu.Compose(_transform)


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
    x = x.to(device, non_blocking=non_blocking)
    y = y.to(device, non_blocking=non_blocking).long()
    return x, y


def pre_transforms(image_size=256):
    return [
        albu.ToFloat(max_value=1.0),
        albu.Resize(image_size, image_size, p=1),
    ]


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.Flip(),
        albu.CoarseDropout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3),
        # albu.HueSaturationValue(p=0.3)
    ]

    return result


def resize_transforms(image_size=256):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    # Converts the image to a square of size image_size x image_size
    result = [
        albu.OneOf([
            random_crop,
            rescale,
            random_crop_big
        ], p=1)
    ]

    return result


def post_transforms(normalization_fn=None):
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    if not normalization_fn:
        normalization_fn = albu.Normalize(max_pixel_value=1.)

    return [normalization_fn, ToTensorCustom()]


def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result

