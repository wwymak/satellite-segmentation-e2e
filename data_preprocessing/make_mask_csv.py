# step 1: make masks from ground truth labels using the solaris cli

import solaris as sol
import numpy as np
import pandas as pd

from pathlib import Path
from fastcore.utils import *

import subprocess

# dataset_name = "AOI_2_Vegas_Train"
dataset_name = "AOI_3_Paris_Train"

data_dir = Path('/media/wwymak/Storage/spacenet')
source_dir = data_dir/dataset_name/"geojson"/"buildings"
reference_dir = data_dir/dataset_name/"RGB-PanSharpen"
output_dir = data_dir/dataset_name/"masks_v2"
output_dir.mkdir(exist_ok=True)


if __name__=="__main__":
    image_ids = ["_".join(x.stem.split('_')[1:]) for x in source_dir.ls()]
    source_file = [source_dir / f"buildings_{x}.geojson" for x in image_ids]
    reference_image = [reference_dir / f"RGB-PanSharpen_{x}.tif" for x in image_ids]
    output_path = [output_dir / f"{x}.tif" for x in image_ids]

    df = pd.DataFrame(data={
        'source_file': source_file,
        'reference_image': reference_image,
        'output_path': output_path,
    })

    df.to_csv(data_dir/dataset_name/"summaryData"/"mask_csv.csv", index=False)

    subprocess.run([
        "make_masks",
        "-t",
        "--batch",
        "--argument_csv",
        str(data_dir/dataset_name/"summaryData"/"mask_csv.csv"),
        "--footprint"
    ])