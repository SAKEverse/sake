# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
##### ------------------------------------------------------------------- #####

def normalize(df, y, base_condition, match_cols):
    """
    Normalize column in dataframe.

    Parameters
    ----------
    df: pd.DataFrame
    y : str, var name, e.g. power
    base_condition : dict, {col:column matching condition, e.g. treatment
                        val: str,  baseline  value, e.g. baseline}
    match_cols: list, column to match for normalization

    Returns
    -------
    df: pd.DataFrame, with normalized y

    """
    
    # create copy and add normalize
    data = df.copy()
    # norm_y = 'norm_' + y
    # data[norm_y] = 0

    # get unique conditions
    combi_list=[]
    for col in match_cols:
        combi_list.append(data[col].unique())

    # normalize to baseline
    combinations = itertools.product(*combi_list)
    pbar = tqdm(desc='Normalizing', total=len(list(combinations)))

    for comb in itertools.product(*combi_list):       
        
        # find unique treatments for each set of conditions
        idx = np.zeros((len(data), len(match_cols)))
        for i, col in enumerate(match_cols):
            idx[:,i] = data[col] == comb[i]
        temp = data[np.all(idx, axis=1)]
        
        # normalize by baseline
        baseline_idx = temp.index[temp[base_condition['col']] == base_condition['val']][0]
        data.at[temp.index, y] = data[y][temp.index] / data[y][baseline_idx]
        
        pbar.update(1)   
    pbar.close()
        
    return data


def normalize_extra_column(df, y, base_condition, match_cols):
    """
    Normalize column in dataframe.

    Parameters
    ----------
    df: pd.DataFrame
    y : str, var name, e.g. power
    base_condition : dict, {col:column matching condition, e.g. treatment
                        val: str,  baseline  value, e.g. baseline}
    match_cols: list, column to match for normalization

    Returns
    -------
    None.

    """
    
    # create copy and add normalize
    data = df.copy()
    norm_y = 'norm_' + y
    data[norm_y] = 0

    # get unique conditions
    combi_list=[]
    for col in match_cols:
        combi_list.append(data[col].unique())

    # normalize to baseline
    combinations = itertools.product(*combi_list)
    pbar = tqdm(desc='Normalizing', total=len(list(combinations)))

    for comb in itertools.product(*combi_list):       
        
        # find unique treatments for each set of conditions
        idx = np.zeros((len(data), len(match_cols)))
        for i, col in enumerate(match_cols):
            idx[:,i] = data[col] == comb[i]
        temp = data[np.all(idx, axis=1)]
        
        # normalize by baseline
        baseline_idx = temp.index[temp[base_condition['col']] == base_condition['val']][0]
        data.at[temp.index, norm_y] = data[y][temp.index] / data[y][baseline_idx]
        
        pbar.update(1)   
    pbar.close()
        
    return data, norm_y