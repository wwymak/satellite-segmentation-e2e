from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix, IoU
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


def log_training_loss(engine):
    print("Epoch[{}] Loss: {:.2f}".format(engine.state.epoch, engine.state.output))

def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch[{}] Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

if __name__=="__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise UserWarning("cuda not available")

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    model, preprocessing_function = unets.unet_resnet('efficientnet-b0')

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    trainer.logger = setup_logger("trainer")
    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)

    num_classes = 1
    cm_metric = ConfusionMatrix(num_classes=num_classes)

    val_metrics = {
        "IoU": IoU(cm_metric),
        "mIoU_bg": mIoU(cm_metric),
        "dice_loss": Loss(loss)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics)
    evaluator.logger = setup_logger("evaluator")
