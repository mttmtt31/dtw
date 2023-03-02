import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from typing import Union

def compute_distance(s:Union[pd.Series, pd.DataFrame], t:Union[pd.Series, pd.DataFrame], distance:str='euclidean')->float:
    """Return the Euclidean distance between two x, y coordinates

    Args:
        s, t (pd.Series): two Dataframes of matching length, each containing values for x and for y coordinate.
        distance (str): distance metrics to use. Defaults to 'euclidean'

    Returns:
        float: total euclidean distance between all the entries of the two DataFrames
    """
    if distance.lower() not in ['euclidean']:
        raise ValueError('Please specify a valid distance measure.')
    if s.empty or t.empty:
        raise ValueError('Series should be non-empty.')
    if len(s) != len(t):
        raise ValueError('Series should be of the same length')
    if distance.lower() == 'euclidean':
        if isinstance(s, pd.DataFrame):
            s.loc[:, "x"] = s.loc[:, "x"] * 1.05
            s.loc[:, "y"] = s.loc[:, "y"] * 0.68
            t.loc[:, "x"] = t.loc[:, "x"] * 1.05
            t.loc[:, "y"] = t.loc[:, "y"] * 0.68
            return np.linalg.norm(s[['x', 'y']].values - t[['x', 'y']].values, axis=1).mean()
        elif isinstance(s, pd.Series):
            return np.sqrt(((s["x"] - t["x"])*1.05)**2 + ((s["y"] - t["y"])*0.68)**2)
        else: 
            raise ValueError('Format type not supported.')
