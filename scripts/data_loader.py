import pandas as pd


def get_metadata():
    csv_path = "./part1/Combined_Labels_DataFrame.csv"
    df = pd.read_csv(csv_path)
    df['path'] = df['path'].apply(lambda path: "./part1/structured_data/" + path)
    return df
