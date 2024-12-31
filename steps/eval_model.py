import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin

from src.evaluation import MSE, R2 , RMSE

@step
def eval_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
    ) -> Tuple[
    Annotated[float, 'R2 Score'],
    Annotated[float, 'RMSE']
    ]:
    """
    Evaluates the model
    """
    try:
        predictions = model.predict(X_test)
        mse = MSE().calculate_scores(y_test, predictions)
        r2 = R2().calculate_scores(y_test, predictions)
        rmse = RMSE().calculate_scores(y_test, predictions)

        return r2 , rmse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e