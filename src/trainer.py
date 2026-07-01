from pathlib import Path
from typing import Dict

import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

from config import (
    CV,
    N_JOBS,
    PARAM_GRID,
    RANDOM_STATE,
    SCORING,
)


class ModelTrainer:

    """
    Train machine learning models using GridSearchCV.
    """

    def __init__(self):

        self.best_model = None

        self.best_params = None

    # ---------------------------------------------------
    # Private
    # ---------------------------------------------------

    def _build_pipeline(
        self,
        scaler: str
    ):

        if scaler == "standard":

            scaler = StandardScaler()

        elif scaler == "minmax":

            scaler = MinMaxScaler()

        else:

            raise ValueError(
                "Scaler must be 'standard' or 'minmax'."
            )

        pipeline = Pipeline(

            [

                ("scaler", scaler),

                (

                    "classifier",

                    SVC(

                        random_state=RANDOM_STATE

                    )

                )

            ]

        )

        return pipeline

    # ---------------------------------------------------
    # Train
    # ---------------------------------------------------

    def train(

        self,

        dataset,

        scaler="standard"

    ):

        pipeline = self._build_pipeline(scaler)

        params = {

            "classifier__C": PARAM_GRID["C"],

            "classifier__kernel": PARAM_GRID["kernel"],

            "classifier__degree": PARAM_GRID["degree"],

            "classifier__gamma": PARAM_GRID["gamma"]

        }

        search = GridSearchCV(

            estimator=pipeline,

            param_grid=params,

            scoring=SCORING,

            cv=CV,

            n_jobs=N_JOBS

        )

        search.fit(

            dataset.X_train,

            dataset.y_train

        )

        self.best_model = search.best_estimator_

        self.best_params = search.best_params_

        return self.best_model

    # ---------------------------------------------------
    # Save
    # ---------------------------------------------------

    def save(

        self,

        path: Path

    ):

        if self.best_model is None:

            raise ValueError(

                "No trained model found."

            )

        joblib.dump(

            self.best_model,

            path

        )

    # ---------------------------------------------------
    # Get Parameters
    # ---------------------------------------------------

    def get_best_params(self):

        return self.best_params