# Module for common exp tracking methods

import os
from pathlib import Path
import mlflow
import torch

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common

MLFLOW_TRACKING_URI = Path('/media/wwymak/Storage/spacenet/experiment_tracking')


def _mlflow_get_output_path():
    return mlflow.get_artifact_uri()


@idist.one_rank_only()
def _mlflow_log_artifact(fp):
    mlflow.log_artifact(fp)


@idist.one_rank_only()
def _mlflow_log_params(params_dict):
    mlflow.log_params(
        {"pytorch version": torch.__version__, "ignite version": ignite.__version__,}
    )
    mlflow.log_params(params_dict)


get_output_path = _mlflow_get_output_path
log_params = _mlflow_log_params
setup_logging = common.setup_mlflow_logging
log_artifact = _mlflow_log_artifact