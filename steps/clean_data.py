import logging

import pandas as pd
from zenml import step


@step
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data by removing missing values
    Args:
        data: pd.DataFrame: Dataframe containing the data
    Returns:
        pd.DataFrame: Dataframe containing the cleaned data
    """
    pass