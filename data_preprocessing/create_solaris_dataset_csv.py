# step 2 create training dataset csv file
import solaris as sol
import numpy as np
import pandas as pd

from pathlib import Path

data_dir = Path('/media/wwymak/Storage/spacenet')

dataset_names = ["AOI_2_Vegas", "AOI_3_Paris"]


if __name__ == "__main__":
    for dataset_name in dataset_names:
        sol.utils.data.make_dataset_csv(
            im_dir=str(data_dir/f"{dataset_name}_Train"/"RGB-PanSharpen"),
            im_ext='tif',
            label_dir=str(data_dir/f"{dataset_name}_Train"/"masks_v2"),
            label_ext='tif',
            output_path=str(data_dir/f"{dataset_name}_Train"/"summaryData"/'dataset_train.csv'),
            stage='train',
            match_re=f"{dataset_name}_img(.*).tif",
            recursive=False,
            verbose=True)

    df = pd.concat([
        pd.read_csv(data_dir/f"{dataset_name}_Train"/"summaryData"/'dataset_train.csv')
        for dataset_name in dataset_names
    ])
    df.to_csv(data_dir/'dataset.csv', index=False)