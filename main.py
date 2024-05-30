import pandas as pd

from data.visualization import *


def update_df_paths(df: pd.DataFrame) -> None:
    df['path'] = df['path'].apply(lambda path: "part1/structured_data/" + path)


def main():
    csv_path = "part1/Combined_Labels_DataFrame.csv"
    df = pd.read_csv(csv_path)
    update_df_paths(df)

    df_grouped_by_label = df.groupby('label')

    df_anger = df_grouped_by_label.get_group('anger')
    df_engaged = df_grouped_by_label.get_group('engaged')
    df_happy = df_grouped_by_label.get_group('happy')
    df_neutral = df_grouped_by_label.get_group('neutral')

    plot_pixel_intensity_distribution('anger', df_anger['path'].tolist())
"""
    for index, row in df_happy.iterrows():
        print(f"{index}:\t{row['label']} {row['path']}")
        
        
    plot_classes_distribution(df_anger, df_engaged, df_happy, df_neutral)
        
    plot_image_dimensions_histogram(df_happy['path'].tolist())
"""


if __name__ == '__main__':
    main()
