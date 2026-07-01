from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    LOG_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


@dataclass
class Dataset:

    X_train: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_test: pd.Series


class DataPreprocessor:

    """
    Data preprocessing utilities.
    """

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe.copy()

    # --------------------------------------------------
    # Log Transform
    # --------------------------------------------------

    def log_transform(self):

        df = self.df.copy()

        for column in LOG_COLUMNS:

            df[column] = np.log1p(df[column])

        return df

    # --------------------------------------------------
    # Split Dataset
    # --------------------------------------------------

    def split(
        self,
        dataframe=None
    ):

        if dataframe is None:

            dataframe = self.df

        X = dataframe.drop(
            TARGET_COLUMN,
            axis=1
        )

        y = dataframe[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            stratify=y,

            random_state=RANDOM_STATE

        )

        return Dataset(

            X_train,

            X_test,

            y_train,

            y_test

        )

    # --------------------------------------------------
    # Prepare Both Versions
    # --------------------------------------------------

    def prepare(self):

        original = self.split()

        log_df = self.log_transform()

        log_version = self.split(log_df)

        return {

            "original": original,

            "log": log_version

        }