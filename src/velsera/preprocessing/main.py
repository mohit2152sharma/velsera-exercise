# Takes a txt file reads it, returns the content
# Takes a directory path and reads the files in it
# Classifies the files depending on their parent directory name
# Collates everything into a single dataframe
# Saves the dataframe to a csv file


import logging

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split as tts

from velsera.utils.files import save_func
from velsera.utils.loggers import log_time

logger = logging.getLogger(__name__)


class Preprocess:
    def __init__(self, cancer_df: DataFrame, non_cancer_df: DataFrame) -> None:
        self.cancer_df = cancer_df
        self.non_cancer_df = non_cancer_df

    @staticmethod
    def remove_redundant_rows(df: DataFrame, cols: list[str]) -> DataFrame:
        """Removes redundant rows from the dataframe"""
        r1 = df.shape[0]
        _df = df.drop_duplicates(subset=cols, keep="first")
        r2 = _df.shape[0]
        logger.info(f"Removed {r1-r2} duplicate rows")
        return _df

    @staticmethod
    def remove_empty_rows(df: DataFrame) -> DataFrame:
        """Removes empty rows from the dataframe"""
        r1 = df.shape[0]
        _df = df.dropna(axis=0, how="all")
        r2 = _df.shape[0]
        logger.info(f"Removed {r1-r2} empty rows")
        return _df

    @staticmethod
    def shuffle(*args: DataFrame) -> DataFrame:
        """Shuffles the dataframes"""
        df = pd.concat(args, ignore_index=True)
        return df.sample(frac=1).reset_index(drop=True)

    @log_time
    def clean(self) -> DataFrame:
        """Cleans the dataframes"""
        cancer_df = self.remove_empty_rows(
            self.remove_redundant_rows(self.cancer_df, ["id"])
        )
        non_cancer_df = self.remove_empty_rows(
            self.remove_redundant_rows(self.non_cancer_df, ["id"])
        )
        logger.info("Cleaned the dataframes")
        return self.shuffle(cancer_df, non_cancer_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from velsera.preprocessing.loader import Loader

    loader = Loader(cancer_dir="Dataset/Cancer", non_cancer_dir="Dataset/Non-Cancer")
    cancer_df, non_cancer_df = loader.load()
    save_func(cancer_df, "Dataset/dfs/cancer.csv")
    save_func(non_cancer_df, "Dataset/dfs/non_cancer.csv")
    preprocess = Preprocess(cancer_df, non_cancer_df)
    # X_train, X_test, y_train, y_test = preprocess.preprocess()
    # print(X_train.head())
    # print(X_test.head())
    # print(y_train.head())
    # print(y_test.head())
