import logging

import pandas as pd
from zenml import step

class IngestData():
    """
    Ingests data from a given path (CSV file)
    """
    def __init__(self , data_path: str):
        """
        Constructor for IngestData
        Args:
            data_path: str: Path to the CSV file
        """
        self.data_path = data_path  
    
    def get_data(self):
        """
        Reads the data from the given path
        Returns:
            pd.DataFrame: Dataframe containing the data
        """
        logging.info(f"Reading data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingests data from a given path (CSV file)
    Args:
        data_path: str: Path to the CSV file
    Returns:
        pd.DataFrame: Dataframe containing the data
    Raises:
        FileNotFoundError: If the file is not found
    """
    try:
        return IngestData(data_path).get_data()
    except FileNotFoundError:
        logging.error(f"File not found at {data_path}")
        raise FileNotFoundError(f"File not found at {data_path}")
    except Exception as e:
        logging.error(f"Error in reading file: {e}")
        raise e