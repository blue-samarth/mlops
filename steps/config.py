from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """
    Model name configuration
    """
    model_name_field: str = 'LinearRegressionModel'