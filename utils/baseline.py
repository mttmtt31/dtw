import pandas as pd
from math import inf
import logging
from tqdm import tqdm
import numpy as np
from utils.commons import compute_distance

def break_ties(event_data:pd.DataFrame)->pd.DataFrame:
    """There may be two events which happen at the same timestamp according to Opta (since time_seconds is an integer).
    Break ties so that the second event is added a +0.1 time_seconds. Continue until all ties are broken.

    Args:
        event_data (pd.DataFrame): event sequence

    Returns:
        pd.DataFrame: event sequence with ties broken
    """
    while event_data['time_seconds'].duplicated().sum():
        mask = event_data['time_seconds'] == event_data['time_seconds'].shift()
        event_data['time_seconds'].loc[mask] = event_data['time_seconds'].loc[mask] + 0.1 

    # round everything to one decimal place to make sure Python doesn't mess up with weird values like 15771.9999999998
    event_data['time_seconds'] = event_data['time_seconds'].round(1)
    
    return event_data

def baseline(event_data:pd.DataFrame, track_data:pd.DataFrame, distance:str="euclidean"):
    if len(track_data) < len(event_data):
        logging.warn('Event sequence is longer than tracking sequence. Maybe you pass them in the wrong order?')
    # break ties
    event_data = break_ties(event_data)
    # find out how many values you can try
    wiggle_room = track_data.iloc[-1].name - (event_data["time_seconds"].iloc[-1] * 10)
    
    # store the best constant to use for synchronisation
    best_t0 = 0
    smallest_error = inf
    for t0 in tqdm(range(int(np.floor(wiggle_room))), desc = 'First alignment...'):

        indices = (event_data['time_seconds'] * 10 + t0).to_list()
        track_data_t0 = track_data.copy()
        # match every event starting from the first
        track_data_t0.loc[indices, "opta_x"] =  event_data[["x", "time_seconds"]].set_index("time_seconds")["x"].to_list()
        track_data_t0.loc[indices, "opta_y"] =  event_data[["y", "time_seconds"]].set_index("time_seconds")["y"].to_list()
        track_data_t0 = track_data_t0[~track_data_t0.isna().any(axis = 1)]
        
        # compute the error        
        # error, max_dist, min_dist = compute_error(event_data_t0)
        error = compute_distance(track_data_t0[["opta_x", "opta_y"]].rename(columns = {'opta_x' : 'x', 'opta_y' : 'y'}), 
                                 track_data_t0[["x", "y"]], 
                                 distance = distance)
        if error < smallest_error:
            best_t0 = t0
            smallest_error = error

    indices = (event_data['time_seconds'] * 10 + best_t0).to_list()
    track_data_t0 = track_data.copy()
    # match every event starting from the first
    track_data_t0.loc[indices, "opta_x"] =  event_data[["x", "time_seconds"]].set_index("time_seconds")["x"].to_list()
    track_data_t0.loc[indices, "opta_y"] =  event_data[["y", "time_seconds"]].set_index("time_seconds")["y"].to_list()

    return track_data_t0.dropna(subset = ['opta_x']), best_t0, smallest_error,