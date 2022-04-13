### --------- IMPORTS --------- ###
import numpy as np
### --------------------------- ###

def exact_match(df_series, match:str):
    """
    Filter pd.series with string and get index

    Parameters
    ----------
    df_series : pd.Series
    match : str, to search the series
    
    Returns
    -------
    idx: bool

    """
    return np.array(df_series == match, dtype = bool)


def contains(df_series, match:str):
    """
    Filter pd.series with string and get index

    Parameters
    ----------
    df_series : pd.Series
    match : str, to search the series
    
    Returns
    -------
    idx: bool

    """
    return np.array(df_series.str.contains(match), dtype = bool)
    
def endswith(df_series, match:str):
    """
    Filter pd.series with string and get index

    Parameters
    ----------
    df_series : pd.Series
    match : str, to search the series
    
    Returns
    -------
    idx: bool

    """
    return np.array(df_series.str.endswith(match), dtype = bool)
    
def startswith(df_series, match:str):
    """
    Filter pd.series with string and get index

    Parameters
    ----------
    df_series : pd.Series
    match : str, to search the series
    
    Returns
    -------
    idx: bool

    """
    return np.array(df_series.str.startswith(match), dtype = bool)