import torch
import numpy as np
import segmentation_models_pytorch as smp


def unet_resnet(encoder):
    # ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['building']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights='imagenet',
        classes=1,
        activation='sigmoid',
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)

    return model, preprocessing_fn