from typing import Callable, List, Tuple
import collections
import os
import torch
import catalyst

from catalyst.dl import utils
from torch import nn
from torch import optim

from catalyst.contrib.nn import DiceLoss, IoULoss
from catalyst.dl import SupervisedRunner

import segmentation_models_pytorch as smp

from catalyst.utils import load_checkpoint, unpack_checkpoint

from catalyst.contrib.nn import RAdam, Lookahead


from pathlib import Path
from models.unets import unet_resnet
from dataflow.dataloaders import get_train_val_loaders
from dataflow.visualisations import tensor_to_rgb
from dataflow.transforms import get_train_augmentation, get_validation_augmentation, prepare_batch_fp32, get_preprocessing