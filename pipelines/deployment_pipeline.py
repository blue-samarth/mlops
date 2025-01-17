import pandas as pd
from pydantic import BaseModel

from zenml import pipeline , step  
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.ingest_data import ingest_df
from steps.eval_model import eval_model


docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseModel):
    """
    Parameters for deployment trigger.
    """
    min_accuracy: float = 0.92

@step
def trigger_deployment( accuracy : float , config: DeploymentTriggerConfig) -> bool:
    """
    Step to check if the model accuracy is above a certain threshold.
    Args:
        accuracy: Model accuracy.
        config: Deployment trigger configuration.
    Returns:
        bool: True if model accuracy is above the threshold.
    """
    return accuracy >= config.min_accuracy

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
                                   min_accuracy : float = 0.92,
                                   workers : int = 1,
                                   timeout : int = DEFAULT_SERVICE_START_STOP_TIMEOUT
                                   ) -> None:
    """
    Continuous deployment pipeline.
    """
    try:
        df : pd.DataFrame = ingest_df()
        X_train , X_test , y_train , y_test = clean_df(df)
        model = train_model(X_train , y_train)
        r2 , mse = eval_model(model , X_test , y_test)
        print(mse)

        deployment_decision = trigger_deployment(
                                accuracy=r2,
                                config=DeploymentTriggerConfig(min_accuracy=min_accuracy)
                                )

        mlflow_model_deployer_step(model=model,
                                deploy_decision=deployment_decision,
                                workers=workers,
                                timeout=timeout
                                )
    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise 