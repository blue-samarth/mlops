from zenml import pipeline
import pandas as pd

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.eval_model import eval_model

@pipeline()
def train_pipeline(data_path : str) -> None:
    df : pd.DataFrame = ingest_df(data_path)
    X_tarin , X_test , y_train , y_test = clean_df(df)
    model = train_model(X_tarin , y_train)
    r2 , mse = eval_model(model , X_test , y_test)
