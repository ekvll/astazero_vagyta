import argparse
import os

import pandas as pd

from ..utils import filename_by_index, filenames_in_directory, print_filenames

path_raw_data = "./data/raw/"
path_preprocessed_data = "./data/preprocessed/"


def load_raw_data(filepath):
    return pd.read_csv(filepath, names=["id", "x", "y", "z"])


def get_id_filter(filename):
    if "PtOut_BoH" in filename:
        return "BoH"
    elif "ML_total" in filename:
        return "ML"
    elif "AZ CrRo PoPr_red_vägyta" in filename:
        return "CR"
    elif "Slope" in filename:
        return "Slope"
    else:
        raise NameError("No filter for this file. Specify an ID filter")


def filter_by_id(df: pd.DataFrame, column_name: str, id: str) -> pd.DataFrame:
    return df[df[column_name].str.contains(id, na=False, case=False)]


def gen_args():
    parser = argparse.ArgumentParser(description="Preprocessing data")
    parser.add_argument("--all", action="store_true", help="Preprocess all files")
    return parser.parse_args()


def flip_df(df):
    y0 = df["y"].iloc[0]
    y1 = df["y"].iloc[-1]
    if y0 > y1:
        print("Flipping dataframe")
        return df[::-1]
    else:
        return df


def extract_last_integer(id_value):
    # Split the id by '-' and take the last part, then convert it to an integer
    return int(id_value.split("-")[-1])


def preprocess(filename: str):
    global path_raw_data, path_preprocessed_data

    print(f"Preprocessing file: {filename}")

    input_filepath = os.path.join(path_raw_data, filename)
    df = load_raw_data(input_filepath)
    id_filter = get_id_filter(filename)

    df_filtered = filter_by_id(df, column_name="id", id=id_filter)

    df_filtered = flip_df(df_filtered)

    if "Slope 8" in filename:
        df_filtered = df_filtered.copy()
        df_filtered["sort_key"] = df_filtered["id"].str.extract(r"-(\d+)$").astype(int)
        df_filtered = df_filtered.sort_values("sort_key").drop(columns="sort_key")

    df_filtered = df_filtered.reset_index(drop=True)

    output_filename = filename.split(".")[0] + ".csv"
    output_filepath = os.path.join(path_preprocessed_data, output_filename)

    df_filtered.to_csv(output_filepath, index=False)
    print(f"Preprocessed data saved to {output_filepath}")


def main(all):
    global path_raw_data

    filenames = filenames_in_directory(path_raw_data)
    print_filenames(filenames)

    # args = gen_args()

    if all == "True":
        for filename in filenames:
            preprocess(filename)
    else:
        filename_index = input("Choose a file index: ")
        filename = filename_by_index(filenames, filename_index)
        preprocess(filename)


if __name__ == "__main__":
    main()
