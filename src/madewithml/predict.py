# madewithml/predict.py
import pandas as pd
from typing import Any, Dict, Iterable, List

import ray

def decode(indices: Iterable[Any], index_to_class: Dict) -> List:
    """Decode indices to labels.

    Args:
        indices (Iterable[Any]): Iterable (list, array, etc.) with indices.
        index_to_class (Dict): mapping between indices and labels.

    Returns:
        List: list of labels.
    """
    return [index_to_class[index] for index in indices]

def predict_tags(df: pd.DataFrame, predictor: ray.train.Predictor, index_to_class: Dict) -> List:
    """Predict tags for input data from a dataframe.

    Args:
        df (pd.DataFrame): dataframe with input features.
        predictor (ray.train.Predictor): loaded predictor from a checkpoint.
        index_to_class (Dict): mapping between indices and labels.

    Returns:
        List: list of predicted labels.
    """
    z = predictor.predict(data=df)["predictions"]
    predictions = decode(z.argmax(1), index_to_class)
    return predictions

@app.command()
def predict_tag():
    pass


if __name__ == "__main__":
    app()


