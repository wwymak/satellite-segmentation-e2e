from solaris.eval.base import Evaluator
from solaris.eval.challenges import get_chip_id
from pathlib import Path
import pandas as pd

def spacenet_buildings_2(prop_csv, truth_csv, conf_field_list=['Confidence'], miniou=0.5, min_area=20, challenge='spacenet_2'):
    """Evaluate a SpaceNet building footprint competition proposal csv.

    Uses :class:`Evaluator` to evaluate SpaceNet challenge proposals.

    Arguments
    ---------
    prop_csv : str
        Path to the proposal polygon CSV file.
    truth_csv : str
        Path to the ground truth polygon CSV file.
    miniou : float, optional
        Minimum IoU score between a region proposal and ground truth to define
        as a successful identification. Defaults to 0.5.
    min_area : float or int, optional
        Minimum area of ground truth regions to include in scoring calculation.
        Defaults to ``20``.
    challenge: str, optional
        The challenge id for evaluation.
        One of
        ``['spacenet_2', 'spacenet_3', 'spacenet_off_nadir', 'spacenet_6']``.
        The name of the challenge that `chip_name` came from. Defaults to
        ``'spacenet_2'``.

    Returns
    -------

    results_DF, results_DF_Full

        results_DF : :py:class:`pd.DataFrame`
            Summary :py:class:`pd.DataFrame` of score outputs grouped by nadir
            angle bin, along with the overall score.

        results_DF_Full : :py:class:`pd.DataFrame`
            :py:class:`pd.DataFrame` of scores by individual image chip across
            the ground truth and proposal datasets.

    """

    evaluator = Evaluator(ground_truth_vector_file=truth_csv)
    evaluator.load_proposal(prop_csv,
                            conf_field_list=conf_field_list,
                            proposalCSV=True
                            )
    results = evaluator.eval_iou_spacenet_csv(miniou=miniou,
                                              iou_field_prefix="iou_score",
                                              imageIDField="ImageId",
                                              min_area=min_area
                                              )
    results_DF_Full = pd.DataFrame(results)

    results_DF_Full['AOI'] = [get_chip_id(imageID, challenge=challenge)
                              for imageID in results_DF_Full['imageID'].values]

    results_DF = results_DF_Full.groupby(['AOI']).sum()

    return results_DF, results_DF_Full


if __name__ == "__main__":
    data_dir = Path('/media/wwymak/Storage/spacenet')
    logdir = data_dir / "experiment_tracking" / "unets" / '2020-08-02-19_512_512_inputs'
    predictions_output_dir = data_dir / "predictions" / "2020-08-02-19_512_512_inputs"
    summary_data = data_dir / 'summary_ids.csv'
    ground_truth_data = data_dir / 'AOI_2_Vegas_Train' / 'summaryData' / 'AOI_2_Vegas_Train_Building_Solutions.csv'
    res = spacenet_buildings_2(
        str(predictions_output_dir / 'train.csv'),
        str(predictions_output_dir / 'train_gt.csv'), miniou=0.5,
        min_area=20, challenge='spacenet_2', conf_field_list=None)
    print(res)
