import logging
from typing_extensions import Annotated
from typing import Tuple

import pandas as pd
from zenml import step

from src.data_cleaning import DataPreProcessingStrategy , DataDivideStrategy , DataCleaning , DataStrategy


@step
def clean_df(data: pd.DataFrame) ->  Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    """
    Cleans the data by removing missing values
    Args:
        data: pd.DataFrame: Dataframe containing the raw data
    Returns:
        pd.DataFrame: X_train: Training data
        pd.DataFrame: X_test: Testing data
        pd.Series: y_train: Training labels
        pd.Series: y_test: Testing labels
    """
    try:
        process_strategy : DataStrategy = DataPreProcessingStrategy()
        data_cleaning : DataStrategy = DataCleaning(data , process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy : DataStrategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data , divide_strategy)

        X_train , X_test , y_train , y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Done")
        return X_train , X_test , y_train , y_test
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e