import logging

import pandas as pd
from zenml import step


@step
def eval_model(data: pd.DataFrame) -> None:
    pass