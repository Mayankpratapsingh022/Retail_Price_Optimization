import mlflow 
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

import mlflow.statsmodels
from typing_extensions import  Annotated
from zenml import step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.logger import get_logger
import pandas as pd
from typing import List, Tuple
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from materializers.custom_materializer import (
    ListMaterializer,
    SKLearnModelMaterializer,
    StatsModelMaterializer,
)

logger = get_logger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker,MLFlowExperimentTracker 
):
    raise RuntimeError(
        'You active stack needs to contain a MLFlow experiment tracker for'
        'this example to work.'
    )

@step(experiment_tracker='mlflow_tracker_mlops',
      settings={'experiment_tracker.mlflow':{'experiment_name':'test_name'}},
      enable_cache=False,output_materializers=[SKLearnModelMaterializer,ListMaterializer])


def sklearn_train(
    X_train: Annotated[pd.DataFrame,'X_train'],
    y_train: Annotated[pd.Series,'y_train']

) -> Tuple[
    Annotated[LinearRegression,'model'],
    Annotated[List[str],'predictors'],
]:
    try:
        mlflow.end_run() 
        with mlflow.start_run() as run:
            mlflow.sklearn.autolog()
            model= LinearRegression()
            model.fit(X_train,y_train)

            predictors = X_train.columns.tolist()
            return model,predictors 
    except Exception as e:
        logger.error(e)
        raise e