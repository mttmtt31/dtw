import pandas as pd
from typing import Union
from math import inf, isinf
import logging
import numpy as np
from tqdm import tqdm
from utils.commons import compute_distance


def refinement(event_data:pd.DataFrame, track_data:pd.DataFrame, distance:str="euclidean", w:Union[int,float]=0.5, increment:int=0, hard_alignment:bool=False):
    """For each event, find the tracking event which minimises the error inside a window. The centre of the window is determined as 
    that tracking index with the same time_seconds as the event plus an (optional) increment which possibly accounts for the systematic 
    shift between the two sequences.

    Args:
        event_data, track_data (pd.DataFrame): Two pd.DataFrame containing values for `x` coordinate, `y` coordinate, and `time_seconds`
        distance (str, optional): distance to use. Defaults to "euclidean".
        w (float, optional): window size (in seconds), defined on every row. This number indicates how many seconds to the left (and symmetrically, to the right) to consider. Defaults to 0.5.
        increment (int, optional): first shifting of the event dataset, usually derived from a baseline. Defaults to 0.
        hard_alignment(bool, optional): once you synchronised event i to track frame j, whether event i+1 can be matched to a track_frame k, with k<j.
    Raises:
        ValueError: when either pd.DataFrame is empty
        ValueError: when the window size is negative.

    Returns:
       pd.DataFrame: synchronisation dataframe
    """
    if len(track_data) < len(event_data):
        logging.warn('Event sequence is longer than tracking sequence. Maybe you pass them in the wrong order?')
    if not len(event_data) or not len(track_data):
        raise ValueError("Both sequences should be both non-empty")
    if w < 0:
        raise ValueError("Window size should be positive") 
    if w > len(track_data):
        raise ValueError("Nuh-uh, you need to select a smaller window. If you don't want to use a window, please use utils.accelerated_dtw instead.")
    if w > 1:
        logging.warn(f'A value of {w} for the window is quite large, as it will consider {int(w*10)} frames on either side.\nIt is advisable to select a smaller window, or to use utils.dtw instead.')
    w = int(w*10)
    event_data = event_data.reset_index()
    track_data = track_data.reset_index()
    # retrieve the lengths of the two series
    r, c = len(event_data), len(track_data)
    # retrieve possible missing values in the tracking sequence
    missing_indices = track_data[track_data[["x", "y"]].isna().any(axis=1)].index
    # initialise the distance matrix, setting everything to infinity
    D = np.full((r, c), inf)
    # initialise two empty lists which will be used to keep track of the best matches
    p, q = [], []
    # initialise an empty list which will keep track of the 'refined' error
    errors = []
    if hard_alignment:
        min_j = 0
    # for every row starting from the seconds
    for i in tqdm(range(r), desc = 'Syncing'):
        # compute the window of values in which the distances are computed.
        # the window is centred in the index of the row in the tracking dataframe with the same time_seconds as the current event you are analysing
        # shifted by increment according to baseline 
        centre_point = track_data[track_data["time_seconds"] == event_data.loc[i]['time_seconds']].index[0] + increment
        # define the boundaries of the window with or without alignment
        # with hard alignment, the left boundary is the maximum between the 'standard' window boundary and the previous event's matching. 
        # (I know right_bound is set to the same value, but improves code readability)
        if hard_alignment:
            left_bound = max(min_j, centre_point - w)
            right_bound = centre_point + w + 1
        else:
            left_bound = centre_point - w
            right_bound = centre_point + w + 1
        for j in range(max(0, left_bound), min(c, right_bound)):
            if j not in missing_indices:
                # compute the distance
                D[i, j] = compute_distance(event_data.loc[i], track_data.loc[j], distance = distance)
        # you scanned all columns, now find the one which minimises the error    
        min_j, error = np.argmin(D[i, :]), np.min(D[i, :])
        if not isinf(error):
            p.append(i)
            q.append(min_j)
            errors.append(error)

    path = np.array(p), np.array(q)

    return path, np.array(errors).mean()

