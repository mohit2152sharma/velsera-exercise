from typing import Literal, cast

from pandas import DataFrame, pandas
from pydantic import BaseModel
from sklearn.model_selection import train_test_split as tts

from velsera.finetuning.prompts import TestingPrompt, TrainingPrompt
from velsera.utils.loggers import log_time


class TrainTestSplit(BaseModel):
    x_train: DataFrame
    x_val: DataFrame
    x_test: DataFrame
    y_train: DataFrame
    y_val: DataFrame
    y_test: DataFrame
    train_split_ratio: float
    seed: int


class Preprocessor:

    @log_time
    @staticmethod
    def train_test_split(
        df: DataFrame,
        target_col,
        train_split_ratio: float = 0.2,
    ) -> TrainTestSplit:
        """Preprocesses the dataframes"""
        X = df.drop(columns=target_col)
        y = df[target_col]

        X_train, X_temp, y_train, y_temp = tts(
            X, y, test_size=train_split_ratio, random_state=42
        )

        X_val, X_test, y_val, y_test = tts(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        return TrainTestSplit(
            x_train=cast(DataFrame, X_train),
            x_val=cast(DataFrame, X_val),
            x_test=cast(DataFrame, X_test),
            y_train=cast(DataFrame, y_train),
            y_val=cast(DataFrame, y_val),
            y_test=cast(DataFrame, y_test),
            train_split_ratio=train_split_ratio,
            seed=42,
        )

    @staticmethod
    def to_finetuner(
        x: DataFrame,
        labels: pandas.Series | None = None,
        prompt: Literal["training", "testing"] = "training",
    ) -> DataFrame:
        """Converts the dataframe to the format required by the finetuner"""

        match prompt:
            case "training":
                if labels is None:
                    raise ValueError("Labels must be provided for training prompts.")
                x["text"] = [
                    TrainingPrompt(title=t, abstract=a, label=l)
                    for t, a, l in zip(x["title"], x["abstract"], labels)
                ]
                return x
            case "testing":
                x["text"] = x.apply(
                    lambda x: TestingPrompt(
                        title=x["title"], abstract=x["abstract"]
                    ).prompt,
                    axis=1,
                )
                return x
            case _:
                raise ValueError(f"Invalid prompt type: {prompt}")
