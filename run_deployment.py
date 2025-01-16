import click
from rich import print

from zenml.integrations.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployer import MLflowModelDeployer
from zenml.integrations.mlflow.services import MLflowDeploymentService

from pipelines.deployment_pipeline import continous_deployement_pipeline

DEPLOY : str = "deploy"
PREDICT : str = "predict"
DEPLOY_AND_PREDICT : str = "deploy_and_predict"

@click.command()
@click.option("--data-path", type=str, required=True, help="Path to the data file.")
@click.option("--config" , "-c",
              type=click.Choice([DEPLOY , PREDICT , DEPLOY_AND_PREDICT]),
              default=DEPLOY_AND_PREDICT,
              help="Deployment configuration"
              "Optionally deploy the model, predict using the model or both."
              "Default is deploy and predict.",
            )
@click.option("--min-accuracy", type=float, default=0.92, help="Minimum accuracy for deployment.")

def run_deployment(data_path : str, config : str, min_accuracy : float):
    mlflow_model_deployer_component = MLflowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        print("Deploying model...")
        continous_deployement_pipeline(data_path=data_path,
                                       workers=3,
                                       timeout=60)

        print("Model deployed successfully.")