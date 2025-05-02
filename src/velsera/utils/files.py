from functools import wraps
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas import DataFrame

from velsera.utils.loggers import log_time


def _func_to_save(df: DataFrame, file_ext: str) -> Callable:
    """Returns a function to save a file based on its extension"""
    mapper = {
        ".csv": df.to_csv,
        ".parquet": df.to_parquet,
        ".json": df.to_json,
    }

    return mapper[file_ext]


def _func_to_load(file_ext: str) -> Callable:
    """Returns a function to load a file based on its extension"""
    mapper = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
    }

    return mapper[file_ext]


def load_df(path: str | Path, *args, **kwargs) -> DataFrame:
    """Loads a dataframe from a csv, parquet or json file"""
    ext = file_ext(path)
    return _func_to_load(ext)(path, *args, **kwargs)


def file_ext(path: str | Path) -> str:
    """Returns the file extension of a path"""
    return Path(path).suffix


@log_time
def save_func(df: DataFrame, path: str | Path) -> None:
    """Saves a dataframe to a csv file"""
    ext = file_ext(path)
    _func_to_save(df, ext)(path, index=False)


def save(file_path: str | Path) -> Callable:
    """Decorator to save a dataframe to a csv, parquet or json file"""

    def first_wrapper(func: Callable) -> Callable:
        @wraps(func)
        def second_wrapper(*args, **kwargs):
            df = func(*args, **kwargs)
            save_func(df, file_path)
            return df

        return second_wrapper

    return first_wrapper
