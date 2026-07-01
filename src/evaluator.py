from pathlib import Path

import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


class Evaluator:

    """
    Evaluate trained machine learning models.
    """

    def __init__(self, model):

        self.model = model

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    def predict(self, X):

        return self.model.predict(X)

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------

    def evaluate(self, X_test, y_test):

        y_pred = self.predict(X_test)

        metrics = {

            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "precision": precision_score(
                y_test,
                y_pred
            ),

            "recall": recall_score(
                y_test,
                y_pred
            ),

            "f1": f1_score(
                y_test,
                y_pred
            ),

            "roc_auc": roc_auc_score(
                y_test,
                y_pred
            )

        }

        report = classification_report(

            y_test,

            y_pred

        )

        cm = confusion_matrix(

            y_test,

            y_pred

        )

        return metrics, report, cm

    # ---------------------------------------------------
    # Save Report
    # ---------------------------------------------------

    def save_report(

        self,

        metrics,

        report,

        save_path

    ):

        with open(

            save_path,

            "w",

            encoding="utf-8"

        ) as file:

            file.write("========== METRICS ==========\n\n")

            for key, value in metrics.items():

                file.write(

                    f"{key}: {value:.4f}\n"

                )

            file.write("\n")

            file.write(

                "========== CLASSIFICATION REPORT ==========\n\n"

            )

            file.write(report)

    # ---------------------------------------------------
    # Save Confusion Matrix
    # ---------------------------------------------------

    def save_confusion_matrix(

        self,

        cm,

        save_path

    ):

        disp = ConfusionMatrixDisplay(

            confusion_matrix=cm

        )

        disp.plot()

        plt.tight_layout()

        plt.savefig(

            save_path,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()