from pathlib import Path
from typing import Dict

import pandas as pd


class EDA:
    """
    Perform Exploratory Data Analysis.
    """

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe.copy()

    # ---------------------------------------------------------
    # Basic Information
    # ---------------------------------------------------------

    def dataset_info(self):

        return self.df.info()

    def shape(self):

        return self.df.shape

    def missing_values(self):

        return self.df.isnull().sum()

    def duplicated_rows(self):

        return self.df.duplicated().sum()

    def statistics(self):

        return self.df.describe()

    def mode(self):

        return self.df.mode()

    # ---------------------------------------------------------
    # Correlation
    # ---------------------------------------------------------

    def pearson_corr(self):

        return self.df.corr(numeric_only=True)

    def spearman_corr(self):

        return self.df.corr(
            method="spearman",
            numeric_only=True
        )

    # ---------------------------------------------------------
    # Outlier Detection
    # ---------------------------------------------------------

    def outliers_iqr(self):

        result = {}

        for column in self.df.columns:

            q1 = self.df[column].quantile(0.25)

            q3 = self.df[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            outliers = self.df[
                (self.df[column] < lower)
                |
                (self.df[column] > upper)
            ][column]

            result[column] = outliers

        return result

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    def generate_report(self):

        report = {

            "shape": self.shape(),

            "missing_values": self.missing_values(),

            "duplicates": self.duplicated_rows(),

            "statistics": self.statistics(),

            "mode": self.mode(),

            "pearson_corr": self.pearson_corr(),

            "spearman_corr": self.spearman_corr()

        }

        return report

    # ---------------------------------------------------------
    # Save Report
    # ---------------------------------------------------------

    def save_report(self, save_path: Path):

        report = self.generate_report()

        with open(save_path, "w", encoding="utf-8") as file:

            file.write("========== DATASET SHAPE ==========\n")

            file.write(str(report["shape"]))

            file.write("\n\n")

            file.write("========== MISSING VALUES ==========\n")

            file.write(str(report["missing_values"]))

            file.write("\n\n")

            file.write("========== DUPLICATED ROWS ==========\n")

            file.write(str(report["duplicates"]))

            file.write("\n\n")

            file.write("========== STATISTICS ==========\n")

            file.write(str(report["statistics"]))

            file.write("\n\n")

            file.write("========== MODE ==========\n")

            file.write(str(report["mode"]))

            file.write("\n\n")

            file.write("========== PEARSON CORRELATION ==========\n")

            file.write(str(report["pearson_corr"]))

            file.write("\n\n")

            file.write("========== SPEARMAN CORRELATION ==========\n")

            file.write(str(report["spearman_corr"]))