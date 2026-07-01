from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    """
    Visualization utilities for exploratory data analysis.
    """

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe.copy()

    # ----------------------------------------------------------
    # Histogram
    # ----------------------------------------------------------

    def plot_histograms(
        self,
        save_path: Path,
        figsize=(16, 10)
    ):

        self.df.hist(
            figsize=figsize,
            bins=20
        )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ----------------------------------------------------------
    # Box Plot
    # ----------------------------------------------------------

    def plot_boxplot(
        self,
        save_path: Path,
        figsize=(16, 8)
    ):

        plt.figure(figsize=figsize)

        self.df.boxplot()

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ----------------------------------------------------------
    # Correlation Heatmap
    # ----------------------------------------------------------

    def plot_correlation(
        self,
        save_path: Path
    ):

        corr = self.df.corr(numeric_only=True)

        plt.figure(figsize=(8, 6))

        plt.imshow(corr)

        plt.colorbar()

        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=90
        )

        plt.yticks(
            range(len(corr.columns)),
            corr.columns
        )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ----------------------------------------------------------
    # Scatter Matrix
    # ----------------------------------------------------------

    def scatter_matrix(
        self,
        target_column: str,
        save_path: Path
    ):

        columns = self.df.columns

        fig, ax = plt.subplots(
            4,
            9,
            figsize=(24, 12)
        )

        used = []

        row = 0
        col = 0

        for feature1 in columns:

            used.append(feature1)

            for feature2 in columns:

                if feature1 != feature2:

                    if feature2 not in used:

                        ax[row, col].scatter(

                            self.df[feature1],

                            self.df[feature2],

                            c=self.df[target_column],

                            s=8

                        )

                        ax[row, col].set_xlabel(feature1)

                        ax[row, col].set_ylabel(feature2)

                        col += 1

                        if col == 9:

                            col = 0

                            row += 1

        plt.tight_layout()

        plt.savefig(

            save_path,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

    # ----------------------------------------------------------
    # Generate All Figures
    # ----------------------------------------------------------

    def generate_all(
        self,
        figure_directory: Path,
        target_column: str
    ):

        self.plot_histograms(

            figure_directory / "histograms.png"

        )

        self.plot_boxplot(

            figure_directory / "boxplot.png"

        )

        self.plot_correlation(

            figure_directory / "correlation_heatmap.png"

        )

        self.scatter_matrix(

            target_column,

            figure_directory / "scatter_matrix.png"

        )