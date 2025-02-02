import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..utils import (
    calc_slope_percentage,
    filename_by_index,
    filenames_in_directory,
    gen_path_img,
    get_xlim_for_filename,
    hide_spines,
    load_preprocessed_data,
    print_filenames,
    profile_data,
)

path_preprocessed_data = "./data/preprocessed/"


def gen_args():
    parser = argparse.ArgumentParser(description="Preprocessing data")
    parser.add_argument("--all", action="store_true", help="Preprocess all files")
    return parser.parse_args()


def get_and_prepare_profiles(filename):
    global path_preprocessed_data

    path_img = gen_path_img(filename)

    input_filepath = os.path.join(path_preprocessed_data, filename)

    df = load_preprocessed_data(input_filepath)
    print(df.head(10))

    df = df[df.y < 280 * 1000]

    xlim = get_xlim_for_filename(filename)
    profiles = profile_data(df, xlim=xlim)

    for ip, p in enumerate(profiles):
        p = p.reset_index(drop=True)
        if p.loc[0, "x"] > 0:
            p = p[::-1].reset_index(drop=True)
            profiles[ip] = p

    df_edge = []
    # fig, ax = plt.subplots()
    for ip, p in enumerate(profiles):
        # if p.y.between(347500, 355000).all():
        if ip == 175:
            p = p.sort_values(by="x")
        data = p.head(5)  # 5 * 50 mm = 250 mm = 25 cm

        slope = calc_slope_percentage(data.x.values, data.z.values)
        slope = np.append(slope, np.nan)

        data = data.assign(slope=slope)
        data.y = data.y.mean()
        # data.loc[:, 'slope'] = slope
        # ax.plot(data.x, data.y, '.', label=f"Profile {ip}")

        df_edge.append(data)
    # ax.set_ylabel('y [mm]')
    # ax.set_xlabel('x [mm]')

    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red", ["#729cff", "#e5e5e5", "#ff6b6b"]
    )  #
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    for p in df_edge:
        scatter = ax.scatter(
            p.x / 1000, p.y / 1000, c=p.slope, norm=norm, cmap=cmap, alpha=0.7
        )

    # Add a colorbar
    ax.set_xlabel("x [m]", fontsize=14)
    ax.set_ylabel("y [m]", fontsize=14)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.4)
    cbar.set_label("Slope [%]", fontsize=14)
    hide_spines(ax)

    # Set the colorbar tick font size
    cbar.ax.tick_params(labelsize=14)

    ax.tick_params(axis="x", labelsize=14)  # X-axis tick labels font size
    ax.tick_params(axis="y", labelsize=14)  # Y-axis tick labels font size

    # Create an inset plot
    ax_inset = inset_axes(ax, width="30%", height="30%", loc="lower left", borderpad=6)

    # Define the zoomed-in region
    y1, y2 = (
        -10,
        279,
    )  # Adjust these values to the region you want to zoom in on

    # Zoomed-in scatter plot
    inset_slopes = []
    for p in df_edge:
        if p.y.between(y1 * 1000, y2 * 1000).all():
            inset_slopes.extend(p.slope.values)
    inset_slopes = np.array(inset_slopes)
    inset_slopes = inset_slopes[~np.isnan(inset_slopes)]
    # print(inset_slopes)
    ax_inset.hist(
        inset_slopes, bins=10, edgecolor="black", linewidth=1, color="#e5e5e5"
    )

    # ax_inset.set_title('In-zoomat')
    ax_inset.set_xlabel("Slope [%]", fontsize=14)
    ax_inset.set_ylabel("Count", fontsize=14)
    # hide_spines(ax_inset)

    ax_inset.tick_params(axis="x", labelsize=14)  # X-axis tick labels font size
    ax_inset.tick_params(axis="y", labelsize=14)  # Y-axis tick labels font size

    # Set custom x-ticks for the histogram
    ax_inset.set_xticks([-4, -2, 0, 2, 4, 6])
    ax_inset.set_xticklabels([-4, -2, 0, 2, 4, 6])

    fig_save_path = os.path.join(path_img, "track_edge")
    os.makedirs(fig_save_path, exist_ok=True)

    plt.savefig(os.path.join(fig_save_path, "heatmap_track_edge.png"), dpi=300)
    plt.close(fig)


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
