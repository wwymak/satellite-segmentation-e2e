from typing import Callable, Optional, Tuple, Union

from pathlib import Path
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, ConcatDataset


from dataflow.datasets import SatelliteSegmentationDataset


def get_train_val_loaders(
    summary_data_filepath: Path,
    train_transforms: Callable,
    val_transforms: Callable,
    train_preprocessing: Optional[Callable] = None,
    val_preprocessing: Optional[Callable] = None,
    batch_size: int = 16,
    num_workers: int = 8,
    limit_train_num_samples: Optional[int] = None,
    limit_val_num_samples: Optional[int] = None,
    drop_empty_images: Optional[bool] = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    summary_data_df = pd.read_csv(summary_data_filepath)
    print(summary_data_df.shape)
    if drop_empty_images:
        summary_data_df = summary_data_df[summary_data_df.has_building]
    print(summary_data_df.shape)

    train_image_filepaths = summary_data_df[summary_data_df.train_val_test =="train"].image_filepath.values
    train_mask_filepaths = summary_data_df[summary_data_df.train_val_test =="train"].mask_filepath.values
    valid_image_filepaths = summary_data_df[summary_data_df.train_val_test =="valid"].image_filepath.values
    valid_mask_filepaths = summary_data_df[summary_data_df.train_val_test =="valid"].mask_filepath.values
    test_image_filepaths = summary_data_df[summary_data_df.train_val_test =="test"].image_filepath.values
    test_mask_filepaths = summary_data_df[summary_data_df.train_val_test =="test"].mask_filepath.values


    train_ds = SatelliteSegmentationDataset(
        image_filepath_list=train_image_filepaths,
        mask_filepath_list=train_mask_filepaths,
        transform=train_transforms, preprocessing=train_preprocessing)
    val_ds = SatelliteSegmentationDataset(
        image_filepath_list=valid_image_filepaths,
        mask_filepath_list=valid_mask_filepaths,
        transform=val_transforms, preprocessing=val_preprocessing)

    test_ds = SatelliteSegmentationDataset(
        image_filepath_list=test_image_filepaths,
        mask_filepath_list=test_mask_filepaths,
        transform=val_transforms, preprocessing=val_preprocessing)

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
        test_ds = Subset(train_ds, train_eval_indices)
    else:
        test_ds = test_ds

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers,
        drop_last=True,)

    val_loader = DataLoader(
        val_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, drop_last=False,
    )

    return train_loader, val_loader, test_loader


# def get_inference_dataloader(
#     image_dir: Path,
#     mask_dir: Optional[Path] = None,
#     summary_data_filepath: Optional[Path] = None,
#     transforms: Optional[Callable] = None,
#     preprocessing: Optional[Callable] = None,
#     batch_size: int = 16,
#     num_workers: int = 8,
#     drop_empty_images: Optional[bool] = True,
#     pin_memory: bool = True,
#     limit_num_samples: Optional[int] = None,
# ) -> DataLoader:
#
#     summary_data_df = pd.read_csv(summary_data_filepath)
#     if drop_empty_images:
#         summary_data_df = summary_data_df[summary_data_df.PolygonWKT_Geo != "POLYGON EMPTY"]
#     image_ids = summary_data_df.ImageId.unique()
#
#     ds = SatelliteSegmentationDataset(
#         image_dir, mask_dir, image_id_list=image_ids,
#         transform=transforms, preprocessing=preprocessing)
#
#     if limit_num_samples is not None:
#         indices = np.random.permutation(len(ds))[:limit_num_samples]
#         dataset = Subset(ds, indices)
#
#     loader = DataLoader(
#         dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=False
#     )
#     return loader