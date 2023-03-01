from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
from utils.commons import compute_distance
from tqdm import tqdm
import pandas as pd
from typing import Union

def dtw(x:pd.Series, y:pd.Series, distance:str="euclidean", warp:int=1, w:Union[int,float]=inf):
    """Computes Dynamic Time Warping (DTW) of two sequences.

    Args:
        x, y (pd.Series): Two pd.Series containing values for `x` coordinate, `y` coordinate, and `time_seconds`
        distance (str, optional): distance to use. Defaults to "euclidean".
        warp (int, optional): how many shifts are computed. Defaults to 1.
        w (Union[int,float], optional): window size, defined on every row. When inf, the whole row will be considered. Defaults to inf.

    Raises:
        ValueError: when either pd.Series is empty
        ValueError: when the window size is smaller than 0 or it is a float and not inf.

    Returns:
        minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path
    """
    if not len(x) or not len(y):
        raise ValueError("Both sequences should be both non-empty")
    if w < 0:
        raise ValueError("Window size should be positive") 
    elif isinstance(w, float) and w < inf:
        raise ValueError(f"Window size can only be an integer or float. {w} is not a valid value.")   
    # retrieve the lengths of the two series
    r, c = len(x), len(y)
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
            centre_point = y[y["time_seconds"] == x.loc[i - 1]['time_seconds']].index[0]
            # set to zero all values inside this window
            D0[i, max(1, centre_point - w + 1):min(c + 1, centre_point + w + 1)] = 0
        # set to zero the value in the top-left corner
        D0[0, 0] = 0
    else:
        # we don't have a window here, 
        # so initialise a matrix of 0s and set to inf the elements of the first row and first column (but the one in the top-left corner)
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    # get rid of the first row and column
    D1 = D0[1:, 1:]  # view
    # for every row
    for i in tqdm(range(r), desc = 'Scanning events... this may take some time'):
        # retrieve the centre point. This will be used to find which values are inside the window previously initialised to zero.
        centre_point = y[y["time_seconds"] == x.loc[i]['time_seconds']].index[0]
        # for every value different from inf on the current row
        for j in range(c):
            if (isinf(w) or (max(0, centre_point - w) <= j <= min(c, centre_point + w))):
                # compute the distance between the i-th event of the first sequence and the j-th event of the second sequence 
                D1[i, j] = compute_distance(x.loc[i], y.loc[j], distance = distance)
    # the cost matrix is now ready, copy it.
    C = D1.copy()
    # for each row
    for i in range(r):
        # retrieve the centre of the window on this row
        centre_point = y[y["time_seconds"] == x.loc[i]['time_seconds']].index[0]
        # find the values inside the window
        if not isinf(w):
            jrange = range(max(0, centre_point - w), min(c, centre_point + w + 1))
        else:
            jrange = range(c)
        # for each value inside the window
        for j in jrange:
            # retrieve the cost for MATCH, INSERTION, DELETION. 
            # Parameter warp tells you how much you can go back/foward in time. 
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j], D0[i, j_k]]
            # add the cost of min(insertion, deletion, match) to the cost
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def accelerated_dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)