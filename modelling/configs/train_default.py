# Basic training configuration
import os
from functools import partial

import cv2
import torch.nn as nn
import torch.optim as optim
# import torch.optim.lr_scheduler as lrs

import segmentation_models_pytorch as smp

from torchvision.models.segmentation import deeplabv3_resnet101
from pathlib import Path

import ignite.distributed as idist

from dataflow.dataloaders import get_train_val_loaders
from dataflow.transforms import prepare_batch_fp32, denormalize, get_train_augmentation, get_validation_augmentation
from models.unets import unet_resnet

# ##############################
# Global configs
# ##############################

seed = 19
device = "cuda"
debug = False

fp16_opt_level = "O2"

num_classes = 1

batch_size = 8 * idist.get_world_size()  # total batch size
val_batch_size = batch_size * 2
num_workers = 12
val_interval = 3
accumulation_steps = 4

val_img_size = 513
train_img_size = 480

# ##############################
# Setup Dataflow
# ##############################

data_dir = Path('/media/wwymak/Storage/spacenet/AOI_3_Paris_Train')
image_dir = data_dir / 'RGB-PanSharpen'
mask_dir = data_dir / 'masks'
summary_data_filepath = data_dir / 'summaryData' / 'AOI_3_Paris_Train_Building_Solutions.csv'

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


train_transforms = get_train_augmentation()
val_transforms = get_validation_augmentation()


train_loader, val_loader, train_eval_loader = get_train_val_loaders(
    image_dir=image_dir,
    mask_dir=mask_dir,
    summary_data_filepath=summary_data_filepath,
    train_transforms=get_train_augmentation(),
    val_transforms=get_validation_augmentation(),
    train_ratio=0.8,
    batch_size=batch_size,
    num_workers=num_workers,
    limit_train_num_samples=100 if debug else None,
    limit_val_num_samples=100 if debug else None,
)

prepare_batch = prepare_batch_fp32

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# ##############################
# Setup models
# ##############################

num_classes = 1
# model = deeplabv3_resnet101(num_classes=num_classes)
model, preprocessing_function = unet_resnet('efficientnet-b0')


def model_output_transform(output):
    return output["out"]


# ##############################
# Setup solver
# ##############################

save_every_iters = len(train_loader)

num_epochs = 100

criterion = smp.utils.losses.DiceLoss()

lr = 3e-4
weight_decay = 5e-4
momentum = 0.9
nesterov = False

# optimizer = optim.SGD(
#     [{"params": model.backbone.parameters()}, {"params": model.classifier.parameters()}],
#     lr=1.0,
#     momentum=momentum,
#     weight_decay=weight_decay,
#     nesterov=nesterov,
# )
optimizer = optim.Adam(
    [
        {"params": model.encoder.parameters(), "lr": 1e-6},
        {"params": model.decoder.parameters(), "lr": lr / 10},
        {"params": model.segmentation_head.parameters(), "lr": lr},
    ],
    lr=lr,
)

le = len(train_loader)


def lambda_lr_scheduler(iteration, lr0, n, a):
    return lr0 * pow((1.0 - 1.0 * iteration / n), a)

#
# lr_scheduler = lrs.LambdaLR(
#     optimizer,
#     lr_lambda=[
#         partial(lambda_lr_scheduler, lr0=lr, n=num_epochs * le, a=0.9),
#         partial(lambda_lr_scheduler, lr0=lr * 10.0, n=num_epochs * le, a=0.9),
#     ],
# )

lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)