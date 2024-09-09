from zenml import pipeline
from zenml.logger import get_logger
from steps.bento_builder import bento_builder
logger = get_logger(__name__)
from zenml.client import Client
from zenml.integrations.constants import BENTOML
print(Client().active_stack.experiment_tracker.get_tracking_uri())
from steps.deployer import bentoml_model_deployer
from zenml.config import DockerSettings
from steps.deployment_trigger_step import deployment_trigger
from steps.data_splitter import combine_data, split_data
from steps.deployer import bentoml_model_deployer
from steps.ingest_data import ingest_data
from steps.process_data import categorical_encoding, feature_engineering
from steps.train_model import sklearn_train

docker_settings = DockerSettings(required_integrations=["BENTOML"])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def training_retail():
    df = ingest_data("retail_prices")
    df_processed = categorical_encoding(df)
    df_transformed = feature_engineering(df_processed)
    X_train, X_test, y_train, y_test = split_data(df_transformed)
    model, predictors = sklearn_train(X_train, y_train)
    rmse = 0.95
    decision = deployment_trigger(accuracy=rmse,min_accuracy=0.80)
    bento = bento_builder(model=model)
    bentoml_model_deployer(bento=bento,deploy_decision=decision)

