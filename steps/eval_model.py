import logging

import pandas as pd
from zenml import step


@step
def eval_model(model : str , data : pd.DataFrame) -> pd.DataFrame | None:
    pass