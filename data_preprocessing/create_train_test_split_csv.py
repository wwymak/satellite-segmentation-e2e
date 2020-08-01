import pandas as pd
from pathlib import Path

data_dir = Path('/media/wwymak/Storage/spacenet')
tilebounds_csvs = list(data_dir.glob("*Train/tile_bounds.csv"))

if __name__=="__main__":

    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    combined = []
    for csv in tilebounds_csvs:
        parent_path = csv.parent
        df = pd.read_csv(csv)
        summary_file = pd.read_csv(list((parent_path / "summaryData").glob("*.csv"))[0])
        non_empty = summary_file[summary_file.PolygonWKT_Geo != "POLYGON EMPTY"]
        df["image_filepath"] = df.image_id.apply(lambda x: str(parent_path / "RGB-PanSharpen"/ f"RGB-PanSharpen_{x}.tif") )
        df["mask_filepath"] = df.image_id.apply(lambda x: str(parent_path / "masks"/ f"mask_{x}.png") )
        df["has_building"] = df.image_id.isin(non_empty.ImageId)
        print(df.has_building.value_counts())

        df = df.sort_values(by=['left', 'bottom', 'right', 'top'])
        df["train_val_test"] = "train"
        df.train_val_test.iloc[int(train_ratio * len(df)): int((train_ratio + valid_ratio) * len(df))] = "valid"
        df.train_val_test.iloc[-int(test_ratio * len(df)):] = "test"
        combined.append(df)

    combined = pd.concat(combined)
    combined.to_csv(data_dir / "summary_ids.csv", index=False)