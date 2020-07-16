from typing import Callable, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, ConcatDataset

import ignite.distributed as idist

from datasets import SatelliteSegmentationDataset


def get_train_val_loaders(
    image_dir: Path,
    mask_dir: Path,
    summary_data_filepath: Path,
    train_transforms: Callable,
    val_transforms: Callable,
    train_ratio: float = 0.8,
    batch_size: int = 16,
    num_workers: int = 8,
    limit_train_num_samples: Optional[int] = None,
    limit_val_num_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    summary_data_df = pd.read_csv(summary_data_filepath)
    image_ids = summary_data_df.ImageId.unique()
    np.random.shuffle(image_ids)

    train_image_ids = image_ids[: int(train_ratio * len(image_ids))]
    val_image_ids = image_ids[int(train_ratio * len(image_ids)):]
    train_ds = SatelliteSegmentationDataset(
        image_dir, mask_dir, image_id_list=train_image_ids,
        transform=train_transforms)
    val_ds = SatelliteSegmentationDataset(
        image_dir, mask_dir, image_id_list=val_image_ids,
        transform=val_transforms)

    if limit_train_num_samples is not None:
        np.random.seed(limit_train_num_samples)
        train_indices = np.random.permutation(len(train_ds))[:limit_train_num_samples]
        train_ds = Subset(train_ds, train_indices)

    if limit_val_num_samples is not None:
        np.random.seed(limit_val_num_samples)
        val_indices = np.random.permutation(len(val_ds))[:limit_val_num_samples]
        val_ds = Subset(val_ds, val_indices)

    # random samples for evaluation on training dataset
    if len(val_ds) < len(train_ds):
        np.random.seed(len(val_ds))
        train_eval_indices = np.random.permutation(len(train_ds))[: len(val_ds)]
        train_eval_ds = Subset(train_ds, train_eval_indices)
    else:
        train_eval_ds = train_ds

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers,
        drop_last=True,)

    val_loader = DataLoader(
        val_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers,
        drop_last=False,
    )

    train_eval_loader = DataLoader(
        train_eval_ds, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, drop_last=False,
    )

    return train_loader, val_loader, train_eval_loader

#
# def get_inference_dataloader(
#     root_path: str,
#     mode: str,
#     transforms: Callable,
#     batch_size: int = 16,
#     num_workers: int = 8,
#     pin_memory: bool = True,
#     limit_num_samples: Optional[int] = None,
# ) -> DataLoader:
#     assert mode in ("train", "test"), "Mode should be 'train' or 'test'"
#
#     get_dataset_fn = get_train_dataset if mode == "train" else get_val_dataset
#
#     dataset = get_dataset_fn(root_path, return_meta=True)
#
#     if limit_num_samples is not None:
#         indices = np.random.permutation(len(dataset))[:limit_num_samples]
#         dataset = Subset(dataset, indices)
#
#     dataset = TransformedDataset(dataset, transform_fn=transforms)
#
#     loader = DataLoader(
#         dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=False
#     )
#     return loader