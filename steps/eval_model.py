import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
import mlflow
from zenml.client import Client

from src.evaluation import MSE , R2 , RMSE

tracker : Client = Client().active_stack.experimental_tracker

@step(experiment_tracker=tracker.name)
def eval_model(model: RegressorMixin, 
               X_test: pd.DataFrame | pd.Series ,
               y_test: pd.DataFrame | pd.Series
               ) -> Tuple[
                Annotated[float, 'R2 Score'],
                Annotated[float, 'RMSE']
                ]:
    """
    Evaluates the model
    Args:
        model: RegressorMixin: Model to evaluate
        X_test: pd.DataFrame: Test features
        y_test: pd.DataFrame: Test labels
    Returns:
        Tuple[float, float]: R2 score and RMSE
    """
    try:
        predictions = model.predict(X_test)
        mse = MSE().calculate_scores(y_test, predictions)
        r2 = R2().calculate_scores(y_test, predictions)
        rmse = RMSE().calculate_scores(y_test, predictions)

        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("RMSE", rmse)

        return r2 , rmse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
