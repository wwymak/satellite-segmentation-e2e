import os
import sys
from catalyst.dl.runner.supervised import SupervisedRunner

from pathlib import Path

current_directory = Path(os.path.dirname(__file__))
sys.path.insert(0,current_directory.parent)
sys.path.extend(['/home/wwymak/udacity-ml-eng/project_satellite_segmentation', '/home/wwymak/udacity-ml-eng/project_satellite_segmentation/modelling'])
from modelling.models.unets import unet_resnet
from post_processing.georeferencing import pixel_mask_to_geodataframe


from tqdm import tqdm

from dataflow.dataloaders import get_train_val_loaders
from dataflow.transforms import get_train_augmentation, get_validation_augmentation

import numpy as np
import pandas as pd
from joblib import Parallel, delayed



data_dir = Path('/media/wwymak/Storage/spacenet')
logdir = data_dir /"experiment_tracking"/"unets"/ '2020-08-02-19_512_512_inputs'
predictions_output_dir = data_dir/"predictions"/"2020-08-02-19_512_512_inputs"
summary_data = data_dir / 'summary_ids.csv'
ground_truth_data = data_dir /'AOI_2_Vegas_Train' /'summaryData' / 'AOI_2_Vegas_Train_Building_Solutions.csv'

ENCODER="efficientnet-b0"

batch_size = 4
num_workers = 12

if __name__ == "__main__":
    print('here')
    model, preprocessing_fn = unet_resnet(ENCODER)
    runner = SupervisedRunner(device="cuda:0", input_key="image", input_target_key="mask", model=model)

    summary_data_df = pd.read_csv(summary_data)
    ground_truth_data_df = pd.read_csv(ground_truth_data)

    train_loader, val_loader, test_loader = get_train_val_loaders(
        summary_data_filepath=summary_data,
        train_transforms=get_train_augmentation(is_display=False, image_size=512, normalisation_fn=None),
        val_transforms=get_validation_augmentation(is_display=False, image_size=512, normalisation_fn=None),
        batch_size=batch_size,
        num_workers=num_workers, drop_empty_images=False
    )

    predictions_all = []
    image_filepaths_all = []
    image_ids_all = []

    for loader in [train_loader, val_loader, test_loader]:
        predictions = np.vstack(list(map(
            lambda x: x["logits"].cpu().numpy(),
            runner.predict_loader(loader=loader, resume=f"{logdir}/checkpoints/best.pth")
        )))
        image_filepaths = loader.dataset.image_filepath_list
        predictions_all.append(predictions)
        image_ids = ["_".join(Path(x).stem.split("_")[1:]) for x in image_filepaths]
        image_filepaths_all.append(image_filepaths)
        image_ids_all.append(image_ids)
    # preds_val = []
    # for prediction, image_filepath in tqdm(list(zip(predictions_all[0], image_filepaths_all[0]))):
    #     gdf = pixel_mask_to_geodataframe(prediction, image_filepath, "8379")
    #     preds_val.append(gdf)
    print('generating preds gdfs')
    preds_val = Parallel(n_jobs=12)(delayed(pixel_mask_to_geodataframe)(prediction, image_filepath, image_id, "8379" ) for prediction, image_filepath, image_id in tqdm(list(zip(predictions_all[1], image_filepaths_all[1],image_ids_all[1]))))
    preds_val = pd.concat([pd.DataFrame(x) for x in preds_val])
    preds_val.to_csv(data_dir/"predictions"/predictions_output_dir/"validation.csv", index=False)
    ground_truth_val = ground_truth_data_df[ground_truth_data_df.ImageId.isin(image_ids_all[1])]
    ground_truth_val.to_csv(data_dir / "predictions" / predictions_output_dir / "validation_gt.csv", index=False)

    preds_train = Parallel(n_jobs=12)(delayed(
        pixel_mask_to_geodataframe)(prediction, image_filepath, image_id,"8379" )
        for prediction, image_filepath, image_id in tqdm(list(zip(
        predictions_all[0], image_filepaths_all[0], image_ids_all[0]))))
    preds_train = pd.concat([pd.DataFrame(x) for x in preds_train])
    preds_train.to_csv(data_dir/"predictions"/predictions_output_dir/"train.csv", index=False)
    ground_truth_train = ground_truth_data_df[ground_truth_data_df.ImageId.isin(image_ids_all[0])]
    ground_truth_train.to_csv(data_dir/"predictions"/predictions_output_dir/"train_gt.csv", index=False)

    preds_test = Parallel(n_jobs=12)(delayed(pixel_mask_to_geodataframe)(prediction, image_filepath, image_id, "8379" ) for prediction, image_filepath, image_id in tqdm(list(zip(predictions_all[2], image_filepaths_all[2], image_ids_all[2]))))
    preds_test = pd.concat([pd.DataFrame(x) for x in preds_test])
    preds_test.to_csv(data_dir/"predictions"/predictions_output_dir/"test.csv", index=False)
    ground_truth_train = ground_truth_data_df[ground_truth_data_df.ImageId.isin(image_ids_all[2])]
    ground_truth_train.to_csv(data_dir/"predictions"/predictions_output_dir/"test_gt.csv", index=False)
