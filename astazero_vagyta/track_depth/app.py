import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from scipy.signal import savgol_filter

from ..qc.app import check_array
from ..utils import (
    calc_energy_removed,
    calc_residual,
    calc_rmse,
    filename_by_index,
    filenames_in_directory,
    find_nearest_index,
    fit_poly_line,
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


def slice_profile(profile, section):
    return profile.loc[(profile.x >= section) & (profile.x < section + 3500)]


def rotate_xy_around_point(x, y, angle_degrees, rotation_point):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )

    translated_x = x - rotation_point[0]
    translated_y = y - rotation_point[1]

    rotated_coords = np.dot(rotation_matrix, np.array([translated_x, translated_y]))

    rotated_x = rotated_coords[0] + rotation_point[0]
    rotated_y = rotated_coords[1] + rotation_point[1]

    return rotated_x, rotated_y


def rotate_profile(x, y):
    poly = fit_poly_line(x, y, degree=1)
    slope = poly.coefficients[0]
    angle_degrees = np.degrees(np.arctan(slope))
    rotation_point = (x.mean(), y.mean())
    return rotate_xy_around_point(x, y, -angle_degrees, rotation_point)


def savgol_smooth(arr, window_length, polyorder):
    return savgol_filter(arr, window_length, polyorder)


def calc_loss_due_to_filter(original, filtered):
    residual = calc_residual(original, filtered)
    rmse = calc_rmse(residual)
    energy_removed = calc_energy_removed(original, residual)
    return rmse, energy_removed


def is_last_element_true(arr):
    return np.sum(arr) == 1 and arr[-1]


def minima_indices_found(minima_indices):
    if not np.any(minima_indices):
        print("No minima found")
        return False
    if is_last_element_true(minima_indices):
        print("Only last element is a minima")
        return False
    return True


def find_minima_indices(arr: np.ndarray):
    minima_indicies = np.r_[True, arr[1:] < arr[:-1]] & np.r_[arr[:-1] < arr[1:], True]
    return minima_indicies


def walk_to_find_local_maxima(y, index):
    left_maxima_idx = index
    while left_maxima_idx > 0 and y[left_maxima_idx - 1] > y[left_maxima_idx]:
        left_maxima_idx -= 1

    right_maxima_idx = index
    while (
        right_maxima_idx < len(y) - 1 and y[right_maxima_idx + 1] > y[right_maxima_idx]
    ):
        right_maxima_idx += 1

    # Check for tolerance on the left side
    if (
        left_maxima_idx > 1
        and y[left_maxima_idx - 2] > y[left_maxima_idx - 1] < y[left_maxima_idx]
    ):
        left_maxima_idx -= 1
        while left_maxima_idx > 0 and y[left_maxima_idx - 1] > y[left_maxima_idx]:
            left_maxima_idx -= 1

    # Check for tolerance on the right side
    if (
        right_maxima_idx < len(y) - 2
        and y[right_maxima_idx + 2] > y[right_maxima_idx + 1] < y[right_maxima_idx]
    ):
        right_maxima_idx += 1
        while (
            right_maxima_idx < len(y) - 1
            and y[right_maxima_idx + 1] > y[right_maxima_idx]
        ):
            right_maxima_idx += 1

    return left_maxima_idx, right_maxima_idx


def get_maxima_values(x, y, left_maxima_idx, right_maxima_idx):
    left_maxima = (x[left_maxima_idx], y[left_maxima_idx])
    right_maxima = (x[right_maxima_idx], y[right_maxima_idx])

    return left_maxima, right_maxima


def calc_track_depth(x, y):
    y_minima_indices = find_minima_indices(y)
    minima_indices_found(y_minima_indices)

    all_minima_x, all_minima_y, all_left_maxima, all_right_maxima = [], [], [], []

    for minima_idx in np.where(y_minima_indices)[0]:
        all_minima_x.append(x[minima_idx])
        all_minima_y.append(y[minima_idx])

        left_maxima_idx, right_maxima_idx = walk_to_find_local_maxima(y, minima_idx)

        left_maxima, right_maxima = get_maxima_values(
            x, y, left_maxima_idx, right_maxima_idx
        )
        all_left_maxima.append(left_maxima)
        all_right_maxima.append(right_maxima)

    indices_to_remove = []
    for i in range(len(all_minima_x)):
        if (
            all_minima_x[i] == all_left_maxima[i][0]
            and all_minima_y[i] == all_left_maxima[i][1]
        ) or (
            all_minima_x[i] == all_right_maxima[i][0]
            and all_minima_y[i] == all_right_maxima[i][1]
        ):
            indices_to_remove.append(i)

    # Remove duplicates in reverse order to avoid index errors
    for i in sorted(indices_to_remove, reverse=True):
        all_minima_x.pop(i)
        all_minima_y.pop(i)
        all_left_maxima.pop(i)
        all_right_maxima.pop(i)

    diffs = []
    for minima_y, left_maxima, right_maxima in zip(
        all_minima_y, all_left_maxima, all_right_maxima
    ):
        left_diff = left_maxima[1] - minima_y
        right_diff = right_maxima[1] - minima_y
        largest_diff = max(left_diff, right_diff)
        diffs.append(round(largest_diff, 2))

    result = {
        "minima_x": all_minima_x,
        "minima_y": all_minima_y,
        "left_maxima": all_left_maxima,
        "right_maxima": all_right_maxima,
        "diffs": diffs,
    }

    return result


def get_and_prepare_profiles(filename: str):
    global path_preprocessed_data

    path_img = gen_path_img(filename)

    path_img_track_depth = os.path.join(path_img, "track_depth")
    os.makedirs(path_img_track_depth, exist_ok=True)

    path_img_track_depth_error = os.path.join(path_img_track_depth, "errors")
    os.makedirs(path_img_track_depth_error, exist_ok=True)

    print(f"Calculating track depth of file: {filename}")

    input_filepath = os.path.join(path_preprocessed_data, filename)
    df = load_preprocessed_data(input_filepath)

    xlim = get_xlim_for_filename(filename)
    profiles = profile_data(df, xlim=xlim)

    section_range = np.arange(-10500, 10501, 3500)

    # result = {}
    result = defaultdict(
        lambda: {
            "section_from": [],
            "section_to": [],
            "section": [],
            "y": [],
            "track_depth_max": [],
        }
    )
    errors = {}
    for profile_index, profile in enumerate(profiles):
        y_mean = round(profile.y.mean(), -2)
        for section_index, section in enumerate(section_range):
            profile_slice = slice_profile(profile, section)

            if profile_slice.empty:
                continue

            if profile_slice.shape[0] <= 20:
                continue

            x = profile_slice.x.values
            # y = profile_slice.y.values
            z = profile_slice.z.values

            check_array(x)
            # check_array(y)
            check_array(z)

            x_rotated, z_rotated = rotate_profile(x, z)

            z_smooth = savgol_smooth(z_rotated, window_length=9, polyorder=3)

            rmse, energy_removed = calc_loss_due_to_filter(z_rotated, z_smooth)

            try:
                # print(section)
                result_track_depth = calc_track_depth(x=x_rotated, y=z_smooth)

                if not result[profile_index]:
                    result[profile_index] = {
                        "section_from": [],
                        "section_to": [],
                        "section": [],
                        "y": [],
                        "track_depth_max": [],
                    }

                # Populate the lists
                result[profile_index]["section_from"].append(section)
                result[profile_index]["section_to"].append(section + 3500)
                result[profile_index]["section"].append((section + section + 3500) / 2)
                result[profile_index]["y"].append(y_mean)
                max_depth = (
                    max(result_track_depth["diffs"])
                    if result_track_depth["diffs"]
                    else 0
                )
                result[profile_index]["track_depth_max"].append(max_depth)

                # result[profile_index] = {
                #     "track_depth_max": max(result_track_depth["diffs"]),
                #     "track_depth_mean": np.mean(result_track_depth["diffs"]),
                #     "track_depth_std": np.std(result_track_depth["diffs"]),
                #     "section_from": section,
                #     "section_to": section + 3500,
                #     "y": y_mean,
                # }

                # print(result[profile_index])

                # plot_section_track_depth(
                #     path_img_track_depth,
                #     section,
                #     x_rotated,
                #     z_smooth,
                #     z_rotated,
                #     result_track_depth["minima_x"],
                #     result_track_depth["minima_y"],
                #     result_track_depth["left_maxima"],
                #     result_track_depth["right_maxima"],
                #     profile_index,
                #     rmse,
                #     energy_removed,
                # )

            except Exception:
                errors[profile_index] = {
                    "section_from": section,
                    "section_to": section + 3500,
                }

                # plot_error(
                #     path_img_track_depth_error,
                #     x,
                #     z,
                #     x_rotated,
                #     z_rotated,
                #     z_smooth,
                #     profile_index,
                #     section,
                # )

    data = []
    for key, value in result.items():
        data.append(value)

    fontsize = 22
    flat_data = []
    for entry in data:
        for section_from, section_to, section, y, track_depth_max in zip(
            entry["section_from"],
            entry["section_to"],
            entry["section"],
            entry["y"],
            entry["track_depth_max"],
        ):
            flat_data.append(
                {
                    "section_from": section_from,
                    "section_to": section_to,
                    "section": section,
                    "y": y,
                    "track_depth_max": track_depth_max,
                }
            )

    df = pd.DataFrame(flat_data)
    df = df.sort_values(by=["y", "section"])

    plot_histogram(
        df,
        "track_depth_max",
        "Maximalt Spårdjup [mm]",
        os.path.join(path_img_track_depth, "track_depth_histogram.png"),
    )

    heatmap_df = df.pivot(index="y", columns="section", values="track_depth_max")
    heatmap_df = heatmap_df[::-1]

    heatmap_df.index = np.asarray(heatmap_df.index.values / 1000).astype(int)

    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red", ["#f7f7f7", "#FFECEC", "#ffcfcf", "#ff6b6b"]
    )

    lower_limit = 0
    lower_mid_limit = 2

    upper_mid_limit = 4
    upper_limit = 6

    boundaries = [
        lower_limit,
        lower_mid_limit,
        upper_mid_limit,
        upper_limit,
        10e6,
    ]
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    chunks = split_into_row_chunks(heatmap_df, chunk_size=50)

    fontsize = 22

    # Plot each chunk
    for idx, chunk in enumerate(chunks):
        mask = chunk.isna()
        # chunk = chunk.dropna(axis=1, how="all")
        if filename == "PtOut_BoH.csv":
            figsize = (14, len(chunk) // 3)
        elif filename == "PtOut_BoH2.csv":
            figsize = (14, len(chunk) // 3)
        elif filename == "PtOut_Slope 20.csv":
            figsize = (17, len(chunk) // 2)
        elif filename == "AZ CrRo PoPr_red_vägyta.csv":
            figsize = (17, len(chunk) // 3)
        else:
            figsize = (14, len(chunk) // 2)

        # print(chunk.head(10))

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            chunk,
            mask=mask,
            cmap=cmap,
            norm=norm,
            annot=True,
            fmt=".0f",
            annot_kws={"fontsize": fontsize - 4},
            cbar_kws={
                # "label": "Lutning [%]",
                "shrink": 0.4,
                "aspect": 20,
                "boundaries": boundaries,
                "ticks": boundaries[:-1],
            },
            linecolor=(0, 0, 0, 0.1),  # Semi-transparent black borders
            linewidths=0.3,
        )
        # Set the colorbar label font size
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel("Spårdjup [mm]", fontsize=fontsize)

        # Set the colorbar tick font size
        cbar.ax.tick_params(labelsize=fontsize)

        y_ticks = ax.get_yticks()

        # Set x and y axis tick label font sizes
        ax.tick_params(axis="x", labelsize=fontsize)  # X-axis tick labels font size
        ax.tick_params(axis="y", labelsize=fontsize)  # Y-axis tick labels font size

        plt.yticks(y_ticks[::4], rotation=0)  # Show every 2nd tick on y-axis

        xticks = gen_xticks(chunk.columns)

        plt.xticks(ticks=[i for i in range(len(xticks))], labels=xticks)

        plt.xlabel("x [m]", fontsize=fontsize)
        plt.ylabel("y [m]", fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(path_img_track_depth, f"track_depth_{idx+1}.png"))
        plt.close()


def gen_xticks(arr):
    ref = [-14000, -10500, -7000, -3500, 0, 3500, 7000, 10500, 14000]
    current_min = 0
    current_max = 0
    for val in arr:
        idx = find_nearest_index(ref, val)
        if ref[idx] < 0:
            if ref[idx] < current_min:
                current_min = ref[idx]
        else:
            if ref[idx] > current_max:
                current_max = ref[idx]

    return np.asarray(list(range(current_min, current_max + 1, 3500))) / 1000


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
