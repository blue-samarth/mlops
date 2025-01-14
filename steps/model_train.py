import logging

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
import mlflow
from zenml.client import Client

from src.model_dev import LinearRegressionModel
from .config import ModelNameConfig

tracker : Client = Client().active_stack.experimental_tracker

@step(experiment_tracker=tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig = ModelNameConfig()
    ) -> RegressorMixin | None:
    """
    Trains the model of the ingested data
    Args:
        X_train: pd.DataFrame: Training data
        X_test: pd.DataFrame: Testing data
        y_train: pd.Series: Training labels
        y_test: pd.Series: Testing labels
    Returns:
        RegressorMixin: Trained model
    """
    try:
        model = None
        if config.model_name_field == 'LinearRegressionModel':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train_model(X_train, y_train)
            return trained_model
        else:
            logging.error(f"Model {config.model_name} not found")
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
