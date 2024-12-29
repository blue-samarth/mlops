import logging
from abc import ABC, abstractmethod , Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Interface for data cleaning strategies.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        """
        Cleans the data according to the strategy.

        :param data: The data to clean.
        :return: The cleaned data.
        """
        pass


class DataPreProcessingStrategy(DataStrategy):
    """"
    Startegy for data preprocessing uses DataStrategy interface
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        """
        Handle the data according to the strategy.

        :param data: The data to handle.
        :return: The preprocessed data.
        """
        try:
            data = data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ], 
            axis=1)
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No Review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop : list[str] = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy for data splitting uses DataStrategy interface
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        """
        Handle the data according to the strategy.

        :param data: The data to handle.
        :return: test and train.
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e
        

class DataCleaning(DataStrategy):
    """
    Class for data cleaning uses DataStrategy interface
    """
    def __init__(self, data : pd.DataFrame ,strategy: DataStrategy):
        self.data = data
        self._strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame , pd.Series]:
        try:
            return self._strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
        
# if __name__ == "__main__":
#     data = pd.read_csv("data/olist_order_items_dataset.csv")
#     data_cleaning = DataCleaning(data, DataPreProcessingStrategy())
#     data_cleaning.handle_data()