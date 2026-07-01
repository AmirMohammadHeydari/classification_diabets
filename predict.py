from pathlib import Path
import argparse

import joblib
import pandas as pd

from config import STANDARD_MODEL


def load_model(model_path: Path):
    """
    Load trained model.
    """
    return joblib.load(model_path)


def predict(model, sample: pd.DataFrame):

    prediction = model.predict(sample)[0]

    probability = None

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(sample)[0]

    return prediction, probability


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--model",

        default=STANDARD_MODEL,

        type=Path,

        help="Path to trained model"

    )

    parser.add_argument(

        "--input",

        required=True,

        type=Path,

        help="CSV file containing one or more samples"

    )

    args = parser.parse_args()

    model = load_model(args.model)

    sample = pd.read_csv(args.input)

    prediction, probability = predict(

        model,

        sample

    )

    print()

    print("=" * 50)

    print("Prediction")

    print("=" * 50)

    print(prediction)

    if probability is not None:

        print()

        print("Probability")

        print(probability)


if __name__ == "__main__":

    main()