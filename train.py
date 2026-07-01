from config import *

from src.data_loader import DataLoader

from src.utils import create_directories

from src.eda import EDA


def main():

    create_directories(

        MODEL_DIR,

        RESULT_DIR,

        FIGURE_DIR,

        REPORT_DIR,

        METRIC_DIR

    )

    loader = DataLoader(DATA_PATH)

    df = loader.load()

    print(df.head())

    print(df.shape)

    eda = EDA(df)

    eda.save_report(

        REPORT_DIR / "eda_report.txt"

    )


if __name__ == "__main__":

    main()