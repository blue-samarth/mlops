from zenml import pipeline
from pandas import DataFrame

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.eval_model import eval_model

@pipeline()
def train_pipeline(data_path : str) -> None:
    df : DataFrame = ingest_df(data_path)
    clean_df(df)
    train_model(df)
    eval_model(df)
