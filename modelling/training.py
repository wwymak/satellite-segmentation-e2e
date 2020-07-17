from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import DiskSaver
from ignite.metrics import ConfusionMatrix, IoU, mIoU
from ignite.utils import setup_logger


from models import unets

import torch

import segmentation_models_pytorch as smp

if __name__=="__main__":

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    model, preprocessing_function = unets.unet_resnet('efficientnet-b0')

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])