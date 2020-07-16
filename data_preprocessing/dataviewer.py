
"""
View tifffiles and masks either layered over each other or side by side with napari
Toggle between grid and overlay view on the napari viewer
"""

import numpy as np
import pandas as pd
from skimage import io
import napari

from pathlib import Path
from PIL import Image

from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
from glob import glob


data_dir = Path('/media/wwymak/Storage/spacenet/AOI_3_Paris_Train')
masks_dir = data_dir / 'masks'

image_dir = data_dir / 'RGB-PanSharpen'
label_dir = data_dir / 'geojson' /'buildings'
summary_df = pd.read_csv(data_dir / 'summaryData' / 'AOI_3_Paris_Train_Building_Solutions.csv')
image_ids = summary_df.ImageId.unique()
image_filenames = [f"RGB-PanSharpen_{image_id}.tif" for image_id in image_ids]
mask_filenames = [f"mask_{image_id}.png" for image_id in image_ids]
label_filenames = [f"buildings_{image_id}.geojson" for image_id in image_ids]
image_filepaths = [image_dir /x for x in image_filenames]

label_filepaths = [label_dir /x for x in label_filenames]
mask_filepaths = [masks_dir /x for x in mask_filenames]

def read_tiffile(filepath):
    image = io.imread(filepath)
    image = (image - image.min()) / (image.max() - image.min())
    return image

def read_mask(filepath):
    return np.array(Image.open(filepath)) * 255


if __name__ == "__main__":
    io.use_plugin('tifffile')
    sample_img = io.imread(image_filepaths[90])
    sample_mask = np.array(Image.open(mask_filepaths[90]))
    lazy_imread = delayed(read_tiffile)  # lazy reader
    lazy_arrays_img = [delayed(read_tiffile)(fn) for fn in image_filepaths]
    lazy_arrays_masks = [delayed(read_mask)(fn) for fn in mask_filepaths]
    dask_arrays_img = [
        da.from_delayed(delayed_reader, shape=sample_img.shape, dtype=sample_img.dtype)
        for delayed_reader in lazy_arrays_img
    ]

    dask_arrays_masks = [
        da.from_delayed(delayed_reader, shape=sample_mask.shape, dtype=sample_mask.dtype)
        for delayed_reader in lazy_arrays_masks
    ]
    # Stack into one large dask.array
    stack_img = da.stack(dask_arrays_img, axis=0)
    stack_masks = da.stack(dask_arrays_masks)

    with napari.gui_qt():

        viewer = napari.view_image(stack_img)
        # viewer.add_image(stack_masks, channel_axis=2, colormap=['red', 'green', 'blue'])
        viewer.add_image(stack_masks)