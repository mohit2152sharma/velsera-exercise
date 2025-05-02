import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas import DataFrame

from velsera.preprocessing.parser import Parser
from velsera.utils.loggers import log_time

logger = logging.getLogger(__name__)


class Loader:
    """Loads a directory of file into a dataframe"""

    def __init__(self, cancer_dir: str, non_cancer_dir: str) -> None:
        self.cancer_dir = cancer_dir
        self.non_cancer_dir = non_cancer_dir
        self.cancer_files = self.get_files(self.cancer_dir, ext=".txt")
        self.non_cancer_files = self.get_files(self.non_cancer_dir, ext=".txt")

    @staticmethod
    def get_files(dir: str, ext: Optional[str] = None) -> list[str]:
        """Returns a list of files in a directory"""
        fs = []
        for file in Path(dir).iterdir():
            if file.is_file():
                if ext:
                    if file.suffix == ext:
                        fs.append(file.resolve())
                else:
                    fs.append(file.resolve())
        logger.info(f"Found {len(fs)} files in {dir}")
        return fs

    @log_time
    def load_files(self, files: list[str], y_label: str) -> DataFrame:
        """Loads a list of files into a dataframe"""
        dfs = []
        for file in files:
            parser = Parser(file)
            content = parser.parse()
            if content:
                df = pd.DataFrame([content.model_dump()])
                dfs.append(df)

        _df = pd.concat(dfs)
        _df["y"] = y_label
        return _df

    @log_time
    def load(self) -> tuple[DataFrame, DataFrame]:
        """Loads a directory of files into a dataframe"""
        cancer_df = self.load_files(self.cancer_files, y_label="cancer")
        non_cancer_df = self.load_files(self.non_cancer_files, y_label="non_cancer")
        return cancer_df, non_cancer_df


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    print("executing loader file")
    loader = Loader(cancer_dir="Dataset/Cancer", non_cancer_dir="Dataset/Non-Cancer")
    cancer_df, non_cancer_df = loader.load()
    print(non_cancer_df.head())
    print(cancer_df.head())
