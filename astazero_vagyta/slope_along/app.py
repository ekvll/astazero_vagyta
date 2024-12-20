import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from ..utils import (
    calc_slope_percentage,
    filename_by_index,
    filenames_in_directory,
    find_nearest_index,
    gen_path_img,
    get_xlim_for_filename,
    load_preprocessed_data,
    plot_histogram,
    print_filenames,
    profile_data,
    split_into_row_chunks,
)

path_preprocessed_data = "./data/preprocessed/"


def gen_args():
    parser = argparse.ArgumentParser(description="Preprocessing data")
    parser.add_argument("--all", action="store_true", help="Preprocess all files")
    return parser.parse_args()


def get_and_prepare_profiles(filename):
    global path_preprocessed_data

    if "Slope 8" in filename:
        lower_limit = -8
        upper_limit = 8
    elif "Slope 12" in filename:
        lower_limit = -12
        upper_limit = 12
    elif "Slope 20" in filename:
        lower_limit = -20
        upper_limit = 20
    else:
        lower_limit = -2
        upper_limit = 2

    path_img = gen_path_img(filename)

    input_filepath = os.path.join(path_preprocessed_data, filename)
    df = load_preprocessed_data(input_filepath)
    xlim = get_xlim_for_filename(filename)
    profiles = profile_data(df, xlim=xlim)

    df_targets = {}
    targets_x = np.arange(-20000, 20000, 500)

    for target_x in targets_x:
        closest_x = []
        closest_y = []
        closest_z = []

        for profile_index, profile in enumerate(profiles):
            if profile.x.min() <= target_x and profile.x.max() >= target_x:
                nearest_index = find_nearest_index(profile.x, target_x)

                closest_x.append(target_x)
                closest_y.append(profile_index * 2000)
                closest_z.append(profile.z.iloc[nearest_index])

            else:
                closest_x.append(np.nan)
                closest_y.append(np.nan)
                closest_z.append(np.nan)

        df_tmp = pd.DataFrame({"x": closest_x, "y": closest_y, "z": closest_z})

        df_targets[target_x] = df_tmp

    df_histogram = pd.DataFrame()
    for key, df in df_targets.items():
        slope = calc_slope_percentage(df.y.values, df.z.values)
        slope = np.append(slope, np.nan)
        df.loc[:, "slope"] = slope
        df_histogram = pd.concat([df_histogram, df])

    lower_mid_limit = lower_limit // 2
    upper_mid_limit = upper_limit // 2

    # Combine the DataFrames into a single DataFrame
    combined_df = pd.concat(df_targets.values(), ignore_index=True)

    # Pivot the DataFrame to create the heatmap data
    heatmap_data = combined_df.pivot_table(index="y", columns="x", values="slope")

    # Create the save path
    img_save_path = os.path.join(path_img, "slope_along_track")
    os.makedirs(img_save_path, exist_ok=True)

    plot_histogram(
        df_histogram,
        "slope",
        "Lutning [%]",
        os.path.join(img_save_path, "slope_histogram"),
    )

    # Reverse the order of rows for the heatmap
    heatmap_data = heatmap_data[::-1]

    heatmap_data.index = heatmap_data.index.values / 1000
    heatmap_data.columns = heatmap_data.columns.values / 1000

    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red", ["#729cff", "#b7ccff", "white", "#ffcfcf", "#ff6b6b"]
    )
    boundaries = [
        -10e6,
        lower_limit,
        lower_mid_limit,
        0,
        upper_mid_limit,
        upper_limit,
        10e6,
    ]
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # Create an annotation matrix that marks values outside the range
    # annot_data = heatmap_data.map(
    #     lambda val: f"{val:.1f}" if val < lower_limit or val > upper_limit else ""
    # )

    annot_data = heatmap_data.map(lambda val: f"{val:.1f}")

    # Split the heatmap data into row-based chunks
    chunks = split_into_row_chunks(heatmap_data, chunk_size=50)

    fontsize = 22

    # Plot each chunk
    for idx, chunk in enumerate(chunks):
        if filename == "PtOut_BoH.csv":
            figsize = (14, len(chunk) // 3)
        elif filename == "PtOut_BoH2.csv":
            figsize = (14, len(chunk) // 3)
        elif filename == "PtOut_Slope 20.csv":
            figsize = (17, len(chunk) // 2)
        else:
            figsize = (14, len(chunk) // 2)

        print(chunk.shape)
        annot_chunk = annot_data.loc[chunk.index, chunk.columns]
        fig = plt.figure(figsize=figsize)

        if "PtOut_Slope" in filename:
            annot_element = annot_chunk
        else:
            annot_element = False

        ax = sns.heatmap(
            chunk,
            cmap=cmap,
            norm=norm,
            annot=annot_element,
            fmt="",
            annot_kws={"fontsize": fontsize - 4},
            cbar_kws={
                # "label": "Lutning [%]",
                "shrink": 0.4,
                "aspect": 20,
                "boundaries": boundaries,
                "ticks": boundaries[1:-1],
            },
            linecolor=(0, 0, 0, 0.1),  # Semi-transparent black borders
            linewidths=0.3,
        )

        # Set the colorbar label font size
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel("Lutning [%]", fontsize=fontsize)

        # Set the colorbar tick font size
        cbar.ax.tick_params(labelsize=fontsize)

        # Get the tick positions for both axes
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()

        # Set x and y axis tick label font sizes
        ax.tick_params(axis="x", labelsize=fontsize)  # X-axis tick labels font size
        ax.tick_params(axis="y", labelsize=fontsize)  # Y-axis tick labels font size

        plt.xticks(x_ticks[::3], rotation=45)  # Show every 2nd tick on x-axis
        plt.yticks(y_ticks[::4])  # Show every 2nd tick on y-axis

        # plt.title(f"Lutning Längs Körbanan (Del {idx + 1}/{len(chunks)})")
        plt.xlabel("x [m]", fontsize=fontsize)
        plt.ylabel("y [m]", fontsize=fontsize)

        # Save the plot for each chunk
        plt.tight_layout()
        plt.savefig(os.path.join(img_save_path, f"slope_along_track_chunk_{idx+1}.png"))
        plt.close(fig)


def main(all):
    global path_preprocessed_data

    filenames = filenames_in_directory(path_preprocessed_data)
    print_filenames(filenames)

    #args = gen_args()

    if all == "True":
        for filename in filenames:
            get_and_prepare_profiles(filename)

    else:
        filename_index = input("Choose a file index: ")
        filename = filename_by_index(filenames, filename_index)
        get_and_prepare_profiles(filename)


if __name__ == "__main__":
    main()
