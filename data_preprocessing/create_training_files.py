import solaris as sol
from solaris.data import data_dir
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from PIL import Image

from pathlib import Path
from fastcore.utils import *

data_dir = Path('/media/wwymak/Storage/spacenet/AOI_3_Paris_Train')
masks_dir = data_dir / 'masks'
masks_dir.mkdir(exist_ok=True)
image_dir = data_dir / 'RGB-PanSharpen'
label_dir = data_dir / 'geojson' /'buildings'
summary_df = pd.read_csv(data_dir / 'summaryData' / 'AOI_3_Paris_Train_Building_Solutions.csv')
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

    gdf = gpd.read_file(str(label_filepath))
    fbc_mask = sol.vector.mask.df_to_px_mask(df=gdf,
                                             channels=['footprint'],
                                             reference_im=str(image_filepath),
                                             burn_value=1)
    Image.fromarray(fbc_mask.squeeze()).save(output_filepath)


if __name__ == "__main__":
    for image_id in tqdm(image_ids):
        # print(image_id)
        parse_create_mask(image_id)
    #
    # Parallel(n_jobs=12)(delayed(parse_create_mask)(image_id) for
    #     image_id in image_ids)


