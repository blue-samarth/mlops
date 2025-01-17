# import numpy as np
import pandas as pd

from zenml import pipeline , step  
from zenml.steps.base_steps import BaseParameters #, Output
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW

# from zenml.integrations.mlflow.model_deployer import MLflowModelDeployer
# from zenml.integrations.mlflow.services import MLflowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.ingest_data import ingest_df
from steps.eval_model import eval_model

docker_settings = DockerSettings(required_integration=[MLFLOW])

class Deployment_trigger_config(BaseParameters):
    """
    Parameters for deployment trigger.
    """
    min_accuracy: float = 0.92

@step
def trigger_deployment( accuracy : float , config: Deployment_trigger_config) -> bool:
    """
    Step to check if the model accuracy is above a certain threshold.
    """
    return accuracy >= config.min_accuracy

@pipeline(enable_cache=True, settings={"docker_settings": docker_settings})
def continous_deployement_pipeline(data_path : str,
                                   workers : int = 1,
                                   timeout : int = DEFAULT_SERVICE_START_STOP_TIMEOUT
                                   ) -> None:
    """
    Continous deployment pipeline.
    """
    df : pd.DataFrame = ingest_df(data_path)
    X_tarin , X_test , y_train , y_test = clean_df(df)
    model = train_model(X_tarin , y_train)
    r2 , mse = eval_model(model , X_test , y_test)

    deployment_decision = trigger_deployment(accuracy=r2)

    mlflow_model_deployer_step(model=model,
                               deployment_decision=deployment_decision,
                               workers=workers,
                               timeout=timeout)