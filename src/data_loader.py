import pandas as pd


class DataLoader:

    """
    Load dataset from csv file.
    """

    def __init__(self, path):

        self.path = path

    def load(self):

        df = pd.read_csv(self.path)

        return df