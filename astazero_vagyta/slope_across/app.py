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
    gen_path_img,
    get_xlim_for_filename,
    load_preprocessed_data,
    plot_histogram,
    print_filenames,
    profile_data,
)

path_preprocessed_data = "./data/preprocessed/"


def gen_args():
    parser = argparse.ArgumentParser(description="Preprocessing data")
    parser.add_argument("--all", action="store_true", help="Preprocess all files")
    return parser.parse_args()


def get_min_max(min_val, max_val):
    return np.floor(min_val / 1000) * 1000, np.ceil(max_val / 1000) * 1000


def slice_array(arr, min_val, max_val) -> np.ndarray:
    result = np.array(arr[(arr >= min_val) & (arr < max_val)])
    indices = np.where((arr >= min_val) & (arr < max_val))
    return result, indices


# Function to create heatmap for a chunk of DataFrame
def create_heatmap(df_chunk, chunk_index, path_heatmaps, num_chunks, filename):
    df_chunk = df_chunk[::-1].reset_index(drop=True)

    # Pivot the DataFrame for mean and std
    pivot_mean = df_chunk.pivot(index="y", columns="x", values="mean")
    pivot_std = df_chunk.pivot(index="y", columns="x", values="std")

    pivot_mean.index = np.asarray(pivot_mean.index.values / 1000).astype(int)
    pivot_mean.columns = pivot_mean.columns.values / 1000

    pivot_std.index = np.asarray(pivot_std.index.values / 1000).astype(int)
    pivot_std.columns = pivot_std.columns.values / 1000

    # Invert the order of the rows
    pivot_mean = pivot_mean.iloc[::-1]
    pivot_std = pivot_std.iloc[::-1]

    # Create a custom colormap with fixed boundaries
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red", ["#729cff", "#b7ccff", "white", "#ffcfcf", "#ff6b6b"]
    )
    boundaries = [-10e6, -2, -1, 0, 1, 2, 10e6]
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # Create a normalization that centers at 0
    # norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    # Create annotations with mean and std, but only if the mean is outside the limits
    annotations = np.empty_like(pivot_mean, dtype=object)
    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            mean_val = pivot_mean.iloc[i, j]
            # std_val = pivot_std.iloc[i, j]

            annotations[i, j] = f"{mean_val:.2f}"

            # # Only annotate if mean is outside the specified limits
            # if mean_val < lower_limit or mean_val > upper_limit:
            #     annotations[i, j] = f"{mean_val:.2f}\n±{std_val:.2f}"
            # else:
            #     annotations[i, j] = ""  # Leave empty if within limits

    # # Create annotations with mean and std
    # annotations = np.empty_like(pivot_mean, dtype=object)
    # for i in range(pivot_mean.shape[0]):
    #     for j in range(pivot_mean.shape[1]):
    #         annotations[i, j] = (
    #             f"{pivot_mean.iloc[i, j]:.2f}\n±{pivot_std.iloc[i, j]:.2f}"
    #         )

    if filename == "PtOut_BoH.csv":
        cell_width = 0.4  # Width of each cell
        cell_height = 0.6  # Height of each cell
    elif filename == "PtOut_BoH2.csv":
        cell_width = 1.0
        cell_height = 0.4
    elif filename == "PtOut_Slope 8.csv" or filename == "PtOut_Slope 12.csv":
        cell_width = 1.3
        cell_height = 0.6
    elif filename == "PtOut_Slope 20.csv":
        cell_width = 1.0
        cell_height = 0.4
    else:
        cell_width = 0.7
        cell_height = 0.4
    fig_width = pivot_mean.shape[1] * cell_width
    fig_height = pivot_mean.shape[0] * cell_height

    fontsize = 24

    # Plot the heatmap
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(
        pivot_mean,
        cmap=cmap,
        norm=norm,
        annot=annotations,
        fmt="",
        cbar_kws={
            # "label": "Lutning Medel [%]",
            "shrink": 0.4,
            "aspect": 20,
            "boundaries": boundaries,
            "ticks": [-2, -1, 0, 1, 2],
        },
        linecolor=(0, 0, 0, 0.1),  # RGBA for semi-transparent black borders
        linewidths=0.3,
    )

    # Set the colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Mean Slope [%]", fontsize=fontsize)

    # Set the colorbar tick font size
    cbar.ax.tick_params(labelsize=fontsize)

    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()

    # Set x and y axis tick label font sizes
    ax.tick_params(axis="x", labelsize=fontsize)  # X-axis tick labels font size
    ax.tick_params(axis="y", labelsize=fontsize)  # Y-axis tick labels font size

    plt.xticks(x_ticks[::3], rotation=45)  # Show every 2nd tick on x-axis
    plt.yticks(y_ticks[::4], rotation=0)  # Show every 2nd tick on y-axis

    plt.xlabel("x [m]", fontsize=fontsize)
    plt.ylabel("y [m]", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig(
        os.path.join(path_heatmaps, f"slope_across_track_{chunk_index + 1}.png")
    )
    plt.close(fig)
    # plt.show()


def get_and_prepare_profiles(filename):
    global path_preprocessed_data

    path_img = gen_path_img(filename)

    path_img_slope_across_track = os.path.join(path_img, "slope_across_track")
    os.makedirs(path_img_slope_across_track, exist_ok=True)

    path_img_slope_across_track_profiles = os.path.join(
        path_img_slope_across_track, "profiles"
    )
    os.makedirs(path_img_slope_across_track_profiles, exist_ok=True)

    print(f"Calculating track depth of file: {filename}")

    input_filepath = os.path.join(path_preprocessed_data, filename)
    df = load_preprocessed_data(input_filepath)
    xlim = get_xlim_for_filename(filename)
    profiles = profile_data(df, xlim=xlim)

    result_y = []
    result_x = []
    result_mean = []
    result_std = []

    for i, p in enumerate(profiles):
        result_y.append(p.y.mean())

        # fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10, 8))
        # fig.suptitle(f"Profil {i+1} (y = {i * 2000} mm)")

        # ax[0].set_title("Mätprofil")
        # ax[0].plot(p.x, p.z, "k.")
        # ax[0].set_ylabel("z [mm]")
        # ax[0].grid(alpha=0.5)
        # hide_spines(ax[0])

        slopes = calc_slope_percentage(p.x.values, p.z.values)

        # ax[1].set_title("Lutning mellan punkter")
        # ax[1].plot(p.x[:-1], slopes, "ko-")

        # ax[1].set_ylabel("Lutning [%]")
        # ax[1].grid(alpha=0.5)
        # hide_spines(ax[1])

        min_val, max_val = get_min_max(p.x[:-1].min(), p.x[:-1].max())
        # print(f"Min: {min_val}, Max: {max_val}")
        # ax[2].set_title("Lutning per 1000 mm")
        tmp_mean = []
        tmp_std = []
        tmp_x = []
        for x_range in np.arange(min_val, max_val, 1000):
            xx, xi = slice_array(p.x[:-1], x_range, x_range + 1000)
            xs = np.array(slopes)[xi]
            if len(xs) == 0:
                continue
            # ax[2].plot(xx, xs)

            mean_slope = np.mean(xs)
            std_slope = np.std(xs)

            # ax[3].errorbar(x_range + 500, mean_slope, yerr=std_slope, fmt="ko")
            tmp_mean.append(mean_slope)
            tmp_std.append(std_slope)
            tmp_x.append(x_range + 500)

        result_x.append(tmp_x)
        result_mean.append(tmp_mean)
        result_std.append(tmp_std)

        # ax[2].grid(alpha=0.5)
        # ax[2].set_ylabel("Lutning [%]")

        # ax[3].set_title("Medelvärde och standardavvikelse per 1000 mm")
        # ax[3].grid(alpha=0.5)
        # ax[3].set_ylabel("Lutning Medel [%]")
        # ax[3].set_xlabel("x [mm]")

        # hide_spines(ax[2])
        # hide_spines(ax[3])

        # fig.tight_layout()
        # plt.savefig(
        #     os.path.join(path_img_slope_across_track_profiles, f"profile_{i+1}.png")
        # )
        # plt.close(fig)

    df_result = pd.DataFrame()
    for y, x, mean, std in zip(result_y, result_x, result_mean, result_std):
        df_tmp = pd.DataFrame({"y": y, "x": x, "mean": mean, "std": std})
        df_result = pd.concat([df_result, df_tmp])

    plot_histogram(
        df_result[(df_result["mean"] > -40) & (df_result["mean"] < 15)],
        "mean",
        "Mean Slope [%]",
        os.path.join(path_img_slope_across_track, "slope_along_track_histogram.png"),
    )

    # Split the DataFrame into chunks of 50 rows
    # chunk_size = 50
    # num_chunks = len(df_result) // chunk_size + (
    #     1 if len(df_result) % chunk_size != 0 else 0
    # )

    # print(df_result.head(10))
    # print(df_result.shape)
    unique_ys = df_result.y.unique()
    num_chunks = 0
    heatmap_idx = 0
    chunks = pd.DataFrame()
    for unique_idx, unique_y in enumerate(unique_ys):
        # print(unique_y)
        df_chunk = df_result.loc[df_result.y == unique_y]
        if num_chunks == 50:
            create_heatmap(
                chunks,
                heatmap_idx,
                path_img_slope_across_track,
                len(unique_ys),
                filename,
            )
            num_chunks = 0
            heatmap_idx += 1
            chunks = pd.DataFrame()
        else:
            # df_chunk = df_chunk[::-1].reset_index(drop=True)
            chunks = pd.concat([chunks, df_chunk])
            num_chunks += 1

    if len(chunks) > 0:
        create_heatmap(
            chunks, heatmap_idx, path_img_slope_across_track, len(unique_ys), filename
        )

    # for chunk_index in range(num_chunks):
    #     df_chunk = df_result.iloc[
    #         chunk_index * chunk_size : (chunk_index + 1) * chunk_size
    #     ]
    #     create_heatmap(df_chunk, chunk_index, path_img_slope_across_track, num_chunks)


def main(all):
    global path_preprocessed_data

    filenames = filenames_in_directory(path_preprocessed_data)
    print_filenames(filenames)

    # args = gen_args()

    if all == "True":
        for filename in filenames:
            get_and_prepare_profiles(filename)

    else:
        filename_index = input("Choose a file index: ")
        filename = filename_by_index(filenames, filename_index)
        get_and_prepare_profiles(filename)


if __name__ == "__main__":
    main()
