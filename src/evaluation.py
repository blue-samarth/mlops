import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error , r2_score , root_mean_squared_error

class Evaluation(ABC):
    """
    Abstract class for evaluation
    """
    @abstractmethod
    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray) -> dict:
        """
        Calculate scores for the model
        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        """
        pass

class  MSE(Evaluation):
    """
    Class to calculate Mean Squared Error
    """
    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray) -> float|int :
        """
        Calculate Mean Squared Error
        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        Returns:
            dict: Dictionary containing the score
        """
        try:
            logging.info('Calculating Mean Squared Error')
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e
        
class  R2(Evaluation):
    """
    Class to calculate R2 score
    """
    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray) -> float :
        """
        Calculate R2 score
        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        Returns:
            r2 score : float : containing the score
        """
        try:
            logging.info('Calculating R2 score')
            r2 = float(r2_score(y_true, y_pred))
            logging.info(f"R2 Score: {r2}")
            print(f"R² Score: {r2:.4f}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 score: {e}")
            raise e
        
class  RMSE(Evaluation):
    """
    Class to calculate Root Mean Squared Error
    """
    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray) -> float :
        """
        Calculate Root Mean Squared Error
        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        Returns:
            rmse : float : containing the score
        """
        try:
            logging.info('Calculating Root Mean Squared Error')
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")