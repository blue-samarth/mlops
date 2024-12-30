import logging

import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

@step
def train_model(self,
                X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series,
                ) -> RegressorMixin:
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
        pass
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e