from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
from utils.commons import compute_distance
from tqdm import tqdm
import pandas as pd
from typing import Union
import logging
import numpy as np

def dtw(event_data:pd.DataFrame, track_data:pd.DataFrame, distance:str="euclidean", warp:int=1, w:Union[int,float]=inf, increment:int=0):
    """Computes Dynamic Time Warping (DTW) of two sequences.

    Args:
        event_data, track_data (pd.DataFrame): Two pd.DataFrame containing values for `x` coordinate, `y` coordinate, and `time_seconds`
        distance (str, optional): distance to use. Defaults to "euclidean".
        warp (int, optional): how many shifts are computed. Defaults to 1.
        w (float, optional): window size (in seconds), defined on every row. This number indicates how many seconds to the left (and symmetrically, to the right) to consider.
        increment (int, optional): first shifting of the event dataset, usually derived from a baseline. Defaults to 0.

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
    w = int(w*10)
    event_data = event_data.reset_index()
    track_data = track_data.reset_index()
    # retrieve the lengths of the two series
    r, c = len(event_data), len(track_data)
    # retrieve possible missing values in the tracking sequence
    missing_indices = track_data[track_data[["x", "y"]].isna().any(axis=1)].index
    # initialise a matrix, of dimension r+1, c+1.
    # The first row and column of this matrix will be initialised to inf, but the value in the top-left corner of the matrix.
    # For each row, the values inside the window will be initialised to zero.
    if not isinf(w):
        # initialise the matrix of dimension r+1 c+1
        D0 = full((r + 1, c + 1), inf)
        # for every row starting from the seconds
        for i in range(1, r + 1):
            # compute the window of values which will be set to 0.
            # the window is centred in the index of the row in the tracking dataframe with the same time_seconds as the current event you are analysing
            # shifted by increment according to baseline and then add 1 because D0 has an extra column at the beginning
            centre_point = track_data[track_data["time_seconds"] == event_data.loc[i - 1]['time_seconds']].index[0] + 1 + increment
            # set to zero all values inside this window
            D0[i, max(1, centre_point - w):min(c + 1, centre_point + w + 1)] = 0
        # set to zero the value in the top-left corner
        D0[0, 0] = 0
    else:
        # we don't have a window here, 
        # so initialise a matrix of 0s and set to inf the elements of the first row and first column (but the one in the top-left corner)
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    # set to infinity all the values corresponding to missing tracking coordinates
    D0[:, missing_indices] = inf
    # get rid of the first row and column
    D1 = D0[1:, 1:]  # view
    # for every row
    for i in tqdm(range(r), desc = 'Scanning events... this may take some time'):
        # retrieve the centre point. This will be used to find which values are inside the window previously initialised to zero.
        centre_point = track_data[track_data["time_seconds"] == event_data.loc[i]['time_seconds']].index[0] + increment
        # for every value different from inf on the current row
        for j in range(c):
            if (isinf(w) or ((max(0, centre_point - w) <= j <= min(c, centre_point + w)) and j not in missing_indices)):
                # compute the distance between the i-th event of the first sequence and the j-th event of the second sequence 
                D1[i, j] = compute_distance(event_data.loc[i], track_data.loc[j], distance = distance)
    # the cost matrix is now ready, copy it.
    C = D1.copy()
    # for every row
    i = 0
    # initialise two variables which keep track of the last finite value
    while i < r:
        # check if the row has values different from infinity
        if isinf(min(D1[i, :])):
            # consider the next row
            i = i + 1
        else:
            # find all columns with values different from nans. These will be the values that will be updated in D1
            for j in [idx for idx, val in enumerate(D1[i, :]) if val < inf]:
                # the 3-value window needs to slide to the left until a finite value is found
                i_prev = i
                j_prev = j
                # consider the previous row until you find one with at least one finite value
                while isinf(min(D0[i_prev, :])):
                    i_prev = i_prev-1
                # consider the previous row until you find a triplet with at least one finite value
                while isinf(min(D0[i_prev, j_prev], D0[i_prev, j_prev + 1], D0[i_prev + 1, j_prev])):
                    j_prev = j_prev-1
                D1[i, j] = D1[i, j] + min((D0[i_prev, j_prev], D0[i_prev, j_prev + 1], D0[i_prev + 1, j_prev]))
            i = i + 1
    if len(event_data) == 1:
        path = zeros(len(track_data)), range(len(track_data))
    elif len(track_data) == 1:
        path = range(len(event_data)), zeros(len(event_data))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def accelerated_dtw(event_data:pd.DataFrame, track_data:pd.DataFrame, distance:str="euclidean", warp:int=1, increment:int=0):
    """Computes Dynamic Time Warping (DTW) of two sequences.

    Args:
        event_data, track_data (pd.DataFrame): Two pd.DataFrame containing values for `x` coordinate, `y` coordinate, and `time_seconds`
        distance (str, optional): distance to use. Defaults to "euclidean".
        warp (int, optional): how many shifts are computed. Defaults to 1.
        increment (int, optional): first shifting of the event dataset, usually derived from a baseline. Defaults to 0.
        
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
    event_data = event_data[["x", "y"]].values
    track_data =     track_data[~track_data[["x", "y"]].isna().any(axis = 1)][["x", "y"]].values
    r, c = len(event_data), len(track_data)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(event_data, track_data, metric=distance)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(event_data) == 1:
        path = zeros(len(track_data)), range(len(track_data))
    elif len(track_data) == 1:
        path = range(len(event_data)), zeros(len(event_data))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def _traceback(D):
    # find the set of rows which have at least one finite value
    valid_rows = np.where(np.isfinite(D).any(axis = 1))[0]
    i = -2
    i_row = valid_rows[i]
    _, j = array(D.shape) - 2
    p, q = [], []
    with tqdm(total=i_row, desc = 'Applying DTW...') as pbar:  
        while (i_row > 0) or (j > 0):                  
            # check if the row contains non-inf value. Otherwise, skip it.
            if isinf(min(D[i_row, :])):
                # consider the previous (valid row)
                i = i-1 
                i_row = valid_rows[i]
                pbar.update(1)
            else:
                # find the first finite previous value
                if isinf(min(D[i_row - 1, :])):
                    # consider the previous (valid row)
                    i = i-1
                    i_row = valid_rows[i]
                else:                        
                    centre, right, bottom = (D[i_row, j], D[i_row, j + 1], D[i_row + 1, j])
                    # if you find a triplet of inf, slide everything to the left until finding values different from infinity
                    if isinf(min(centre, right, bottom)):
                        j = j-1                        
                    else:
                        tb = argmin((centre, right, bottom))
                        # matching
                        if tb == 0:
                            # move diagonally
                            i -= 1
                            i_row = valid_rows[i]
                            j -= 1
                            pbar.update(1)
                        # deletion
                        elif tb == 1:
                            # move up
                            i -= 1
                            i_row = valid_rows[i]
                        # insertion
                        else: 
                            # move to the left
                            j -= 1
                            pbar.update(1)
                        p.insert(0, i_row)
                        q.insert(0, j)
    return array(p), array(q)