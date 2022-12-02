import pandas as pd


class DatabaseHandler:

    def write(self, df):
        # save the data to csv
        df.to_csv('reddit.csv', index=False)

    def read(self):
        try:
            df = pd.read_csv('reddit.csv')
            return df
        except FileNotFoundError:
            return pd.DataFrame()