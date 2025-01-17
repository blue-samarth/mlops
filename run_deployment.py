from typing import cast
import click
from rich import print

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService

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
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        print("Deploying model...")
        continous_deployement_pipeline(data_path=data_path,
                                       min_accuracy=min_accuracy,
                                       workers=3,
                                       timeout=60)

        print("Model deployed successfully.")
    if predict:
        print("Predicting using the model...")
        # mlflow_service = MLflowDeploymentService(tracking_uri=get_tracking_uri())
        # mlflow_service.predict()
        print("Prediction done.")
    print(
        "You can now run"
        f"[italic green]mlflow ui --backend-store-uri {get_tracking_uri()}[/italic green]"
        "\nto see the model in the MLflow UI.\n"
        "We can find our tracked runs within the `mlflow_pipeline` experiment."
    )

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continous_deployement_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service : MLflowDeploymentService = cast(MLflowDeploymentService, existing_services[0])
        print("Model server found.")
        if service.is_running():
            print(
                "The MLFLOW prediction service is running as daemon"
                "process services and accepts interface requests at"
                f"{service.prediction_uri}"
                "To stop the service, run"
                f"[italic green]`zenml model-deployer model delete`[/italic green]"
                f"{str(service.uuid)}"
            )
        elif service.is_failed():
            print("The service failed to start."
                  f"last state: {service.status.state.value}"
                  f"last error: {service.status.last_error}"
                  )
            
        else:
            print(
                "The MLFLOW prediction service is not running."
            )

if __name__ == "__main__":
    data_path = "data/olist_customers_dataset.csv"
    run_deployment(data_path=data_path)