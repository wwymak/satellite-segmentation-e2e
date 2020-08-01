
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


data_dir = Path('/media/wwymak/Storage/spacenet')
tilesets_dir = list(data_dir.glob("*Train"))
masks_dir = [x / 'masks' for x in tilesets_dir]
image_dir = [x / 'RGB-PanSharpen' for x in tilesets_dir]
label_dir = [x / 'geojson' / 'buildings' for x in tilesets_dir]
summary_df = pd.read_csv(data_dir / 'summary_ids.csv')
image_ids = summary_df.image_id.unique()

image_filepaths = summary_df.image_filepath
mask_filepaths = summary_df.mask_filepath

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