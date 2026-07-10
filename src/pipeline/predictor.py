import pandas as pd

from src.pipeline.model_loader import load_selected_model


def predict(model_name, input_data):
    """
    Predict using the selected trained model.

    Parameters
    ----------
    model_name : str
        Name of the trained model.

    input_data : pandas.DataFrame
        Processed feature dataframe.

    Returns
    -------
    predictions
    """

    model = load_selected_model(model_name)

    predictions = model.predict(input_data)

    return predictions