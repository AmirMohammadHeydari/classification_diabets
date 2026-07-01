from pathlib import Path
import joblib

from config import *
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.evaluator import Evaluator


def evaluate_model(model_path, dataset, name):

    model = joblib.load(model_path)

    evaluator = Evaluator(model)

    metrics, report, cm = evaluator.evaluate(

        dataset.X_test,

        dataset.y_test

    )

    evaluator.save_report(

        metrics,

        report,

        REPORT_DIR / f"{name}.txt"

    )

    evaluator.save_confusion_matrix(

        cm,

        FIGURE_DIR / f"{name}_cm.png"

    )

    print(name)

    print(metrics)


def main():

    df = DataLoader(DATA_PATH).load()

    datasets = DataPreprocessor(df).prepare()

    evaluate_model(

        STANDARD_MODEL,

        datasets["original"],

        "standard"

    )

    evaluate_model(

        MINMAX_MODEL,

        datasets["original"],

        "minmax"

    )

    evaluate_model(

        LOG_STANDARD_MODEL,

        datasets["log"],

        "log_standard"

    )

    evaluate_model(

        LOG_MINMAX_MODEL,

        datasets["log"],

        "log_minmax"

    )


if __name__ == "__main__":

    main()