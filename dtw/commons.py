import numpy as np
import pandas as pd

def compute_distance(s:pd.Series, t:pd.Series, distance:str='euclidean')->float:
    """Return the Euclidean distance between two x, y coordinates

    Args:
        s, t (pd.Series): two series, each containing values for x and for y coordinate.
        distance (str): distance metrics to use. Defaults to 'euclidean'

    Returns:
        float: _description_
    """
    if distance.lower() not in ['euclidean']:
        raise ValueError('Please specify a valid distance measure.')
    if distance.lower() == 'euclidean':
        return np.sqrt((s["x"] - t["x"])**2 + (s["y"] - t["y"])**2)