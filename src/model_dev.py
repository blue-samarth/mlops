import logging
from abc import ABC, abstractmethod
from typing import Tuple 
# from typing_extensions import Annotated

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator


class Model(ABC):
    @abstractmethod
    def train_model(self, X_train: Tuple, y_train: Tuple) -> None:
        pass

    # @abstractmethod
    # def predict(self, X_test: Tuple) -> Tuple:
    #     pass


class LinearRegressionModel(Model):
    """
    This class is used to train a Linear Regression model
    """
    def train_model(self, X_train, y_train , **kwargs) -> BaseEstimator|None:
        """
        Trains the Linear Regression model
        Args:
            X_train: Tuple: Training data
            y_train: Tuple: Training labels
            **kwargs: Any additional arguments to be passed to the model
        Returns:
            LinearRegression: Trained model
        """
        try:
            reg  = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Trained")
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e
        