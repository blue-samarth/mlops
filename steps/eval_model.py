import logging

import pandas as pd
from zenml import step


@step
def eval_model(model : callable , data) -> None:
    pass