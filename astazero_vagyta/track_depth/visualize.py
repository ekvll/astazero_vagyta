import os

import matplotlib.pyplot as plt
import numpy as np

from ..utils import hide_spines


def plot_section_track_depth(
    path_img,
    section,
    x,
    y,
    y_rotated,
    all_minima_x,
    all_minima_y,
    all_left_maxima,
    all_right_maxima,
    profile_index,
    rmse,
    energy_removed,
):
    fig, ax = plt.subplots(
        nrows=len(all_minima_x),
        ncols=1,
        figsize=(14, 4 * len(all_minima_x)),
        sharex=True,
        sharey=True,
    )

    ax[0].set_title(
        f"Profil: {profile_index}\nFiltrerat: RMSE = {rmse:.2f}, Energi = {energy_removed:.2f}%\n\nMinima: 1/{len(all_minima_x)}"
    )

    for ax_idx, (left_maxima, right_maxima) in enumerate(
        zip(all_left_maxima, all_right_maxima)
    ):

        if ax_idx != 0:
            ax[ax_idx].set_title(f"Minima: {ax_idx+1}/{len(all_minima_x)}")

        plot_profile_with_annotations(
            ax[ax_idx],
            x,
            y,
            y_rotated,
            all_minima_x[ax_idx],
            all_minima_y[ax_idx],
            left_maxima,
            right_maxima,
        )

    fig.tight_layout()

    output_img_name = f"profile-{profile_index}_{section}-{section+3500}.png"
    fig.savefig(os.path.join(path_img, output_img_name))
    plt.close(fig)


def plot_profile_with_annotations(
    ax, x, y, y_rotated, minima_x, minima_y, left_maxima, right_maxima
):

    ax.plot(
        x,
        y_rotated,
        marker="None",
        linestyle="-",
        color="grey",
        label="Mätprofil Roterad",
    )

    ax.plot(
        x, y, marker=".", linestyle="None", color="black", label="Mätprofil Filtrerad"
    )

    ax.plot(
        minima_x,
        minima_y,
        marker="o",
        markersize=10,
        linestyle="None",
        color="green",
        label="Minima",
    )

    ax.plot(
        left_maxima[0],
        left_maxima[1],
        marker="s",
        markersize=10,
        linestyle="None",
        color="blue",
        label="Maxima Vänster",
    )

    ax.plot(
        right_maxima[0],
        right_maxima[1],
        marker="s",
        markersize=10,
        linestyle="None",
        color="red",
        label="Maxima Höger",
    )

    annotate_differences(ax, minima_y, left_maxima, right_maxima)
    hide_spines(ax)


def annotate_differences(ax, minima_y, left_maxima, right_maxima):
    left_diff = left_maxima[1] - minima_y
    right_diff = right_maxima[1] - minima_y
    horizontal_distance = right_maxima[0] - left_maxima[0]

    if left_diff > right_diff:
        largest_diff = left_diff
        annotate_point = left_maxima
    else:
        largest_diff = right_diff
        annotate_point = right_maxima

    annotate_text = f"dz: {largest_diff:.2f}\ndx: {np.abs(horizontal_distance):.2f}"
    ax.annotate(
        annotate_text,
        annotate_point,
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )


def plot_error(
    path_img_track_depth_error,
    x,
    z,
    x_rotated,
    z_rotated,
    z_smooth,
    profile_index,
    section,
):
    fig_e, ax_e = plt.subplots(nrows=2)

    ax_e[0].plot(x, z, "k.-")

    ax_e[0].set_title(
        f"Error i profil {profile_index} sektion {section}-{section+3500}\nMätprofil Orginal"
    )

    ax_e[0].set_xlabel("x [mm]")
    ax_e[0].set_ylabel("z [mm]")

    ax_e[1].set_title("Processad Mätprofil")
    ax_e[1].plot(x_rotated, z_rotated, "k.-", label="Mätprofil Roterad")
    ax_e[1].set_xlabel("x [mm]")
    ax_e[1].set_ylabel("z [mm]")

    ax_e[1].plot(x_rotated, z_smooth, "r.-", label="Mätprofil Filtrerad")
    ax_e[1].legend()

    fig_e.tight_layout()

    output_img_name = f"profile-{profile_index}_{section}-{section+3500}.png"

    fig_e.savefig(os.path.join(path_img_track_depth_error, output_img_name))
    plt.close(fig_e)
