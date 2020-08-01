import solaris as sol
import rasterio
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from PIL import Image

from pathlib import Path
from fastcore.utils import *

data_dir = Path('/media/wwymak/Storage/spacenet/AOI_3_Paris_Train')
# data_dir = Path('/media/wwymak/Storage/spacenet/AOI_2_Vegas_Train')

masks_dir = data_dir / 'masks'
masks_dir.mkdir(exist_ok=True)
image_dir = data_dir / 'RGB-PanSharpen'
label_dir = data_dir / 'geojson' /'buildings'
summary_df = pd.read_csv(list((data_dir/'summaryData').glob("*.csv"))[0])
image_ids = summary_df.ImageId.unique()
image_filenames = [f"RGB-PanSharpen_{image_id}.tif" for image_id in image_ids]
mask_filenames = [f"mask_{image_id}.tif" for image_id in image_ids]
label_filenames = [f"buildings_{image_id}.geojson" for image_id in image_ids]
image_filepaths = [image_dir /x for x in image_filenames]

label_filepaths = [label_dir /x for x in label_filenames]
mask_filepaths = [masks_dir /x for x in mask_filenames]


def parse_create_mask(image_id):
    image_filepath = image_dir /f"RGB-PanSharpen_{image_id}.tif"
    label_filepath = label_dir / f"buildings_{image_id}.geojson"
    output_filepath = masks_dir /f"mask_{image_id}.png"

    dataset = rasterio.open(str(image_filepath))
    tile_bounds = dataset.bounds
    gdf = gpd.read_file(str(label_filepath))
    fbc_mask = sol.vector.mask.df_to_px_mask(df=gdf,
                                             channels=['footprint'],
                                             reference_im=str(image_filepath),
                                             burn_value=1)
    Image.fromarray(fbc_mask.squeeze()).save(output_filepath)
    return image_id, tile_bounds.left, tile_bounds.bottom, tile_bounds.right, tile_bounds.top


if __name__ == "__main__":
    # we also want to save the boundary of each tile. In the train/validation/test splits
    # it is better to have continous regions of tiles, since if we mix up the train/validatino/test
    # tiles it is much easier for the net to learn how to predict the validation/test tiles
    # and doesn't give such an accurate picture of overall perf
    tile_bounds_metadata = []
    for image_id in tqdm(image_ids):
        image_id, left, bottom, right, top = parse_create_mask(image_id)
        tile_bounds_metadata.append({
            "image_id": image_id,
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top
        })
    pd.DataFrame(tile_bounds_metadata).to_csv(data_dir / "tile_bounds.csv", index=False)
    #
    # Parallel(n_jobs=12)(delayed(parse_create_mask)(image_id) for
    #     image_id in image_ids)


