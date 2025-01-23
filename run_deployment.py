from typing import cast
import click
from rich import print

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService

from pipelines.deployment_pipeline import continuous_deployment_pipeline

# constants for deployment configuration
DEPLOY : str = "deploy"
PREDICT : str = "predict"
DEPLOY_AND_PREDICT : str = "deploy_and_predict"

@click.command()
@click.option("--config" , "-c",
              type=click.Choice([DEPLOY , PREDICT , DEPLOY_AND_PREDICT]),
              default=DEPLOY_AND_PREDICT,
              help="Deployment configuration"
              "Optionally deploy the model, predict using the model or both."
              "Default is deploy and predict.",
            )
@click.option(
    "--min-accuracy", 
    type=float, 
    default=0, 
    help="Minimum accuracy for deployment.")

def run_deployment(config : str, min_accuracy : float) -> None:
    """
    Run the MLFlow deployment pipeline.
    Args:
        config: Deployment configuration.
        min_accuracy: Minimum accuracy for deployment.
    """
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    if not mlflow_model_deployer_component:
        print("No active MLFlow model deployer found.")
        return
    
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        print("Deploying model...")
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60
          )
        print("Model deployed successfully.")

    if predict:
        print("Predicting using the model...")
        print("Prediction done.")
    print(
        "You can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}"
        "[/italic green]\n ...to inspect your experiment runs within the MLflow"
        " UI.\nYou can find your runs tracked within the "
        "`mlflow_pipeline` experiment. There you'll also be able to "
        "compare two or more runs.\n\n"
    )

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service : MLFlowDeploymentService = cast(MLFlowDeploymentService, existing_services[0])
        print("Model server found.")
        print(f"Service status: {service.status.state.value}")  
        print(f"Prediction URI: {service.prediction_url}")      
        print(f"Service UUID: {service.uuid}")
        if service.is_running:
            print(
                "The MLFLOW prediction service is running as daemon"
                "process services and accepts interface requests at"
                f"{service.prediction_uri}"
                "To stop the service, run"
                f"[italic green]`zenml model-deployer model delete`[/italic green]"
                f"{str(service.uuid)}"
            )
        elif service.is_failed:
            print("The service failed to start."
                  f"last state: {service.status.state.value}"
                  f"last error: {service.status.last_error}"
                  )
            
        else:
            print(
                "The MLFLOW prediction service is not running."          
            )
            print(f"Current service state: {service.status.state.value}")
    else:
        print("No model server found.")
        print(
            "You can start the service by running"
            f"[italic green]`zenml model-deployer models start`[/italic green]."
        )


if __name__ == "__main__":
    run_deployment()
  