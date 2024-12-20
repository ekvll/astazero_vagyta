import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

path_img_root = "./data/img/"
os.makedirs(path_img_root, exist_ok=True)


def filenames_in_directory(path: str) -> list[str]:
    """Return a list of filenames in a directory.

    Args:
        path (str): Path to the directory

    Returns:
        list[str]: List of filenames in the directory
    """
    return os.listdir(path)


def print_filenames(filenames: list[str]) -> None:
    """Print filenames with their corresponding index.

    Args:
        filenames (list[str]): List of filenames
    """
    print("Index\tFilename")
    for i, filename in enumerate(filenames):
        print(f"{i}\t{filename}")


def filename_by_index(filenames: list[str], index: int) -> str:
    """Choose a filename from a list of filenames by index.

    :param filenames: List of filenames
    :type filenames: list[str]
    :param index: Chosen index
    :type index: int
    :raises Exception: If the index is out of range
    :return: Selected filename
    :rtype: str
    """
    if isinstance(index, str):
        index = int(index)
    try:
        return filenames[index]
    except IndexError:
        raise Exception(f"Index {index} is out of range")


def load_preprocessed_data(filepath: str) -> pd.DataFrame:
    """Load preprocessed data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file

    Example
    -------
    >>> df = load_preprocessed_data("data/preprocessed/ML_total_1.csv")

    Returns
    -------
    pd.DataFrame
        DataFrame containing the preprocessed data
    """
    return pd.read_csv(filepath)


def profile_data(
    df: pd.DataFrame, xlim: int, threshold: int = 700, plot: bool = False
) -> list[pd.DataFrame]:
    """
    Generate profiles from a given DataFrame with x, y, z columns.

    Args:
        df (pd.DataFrame): DataFrame containing 'x', 'y', and 'z' columns representing profile data.
        xlim (int): Minimum x-coordinate threshold. Drops a profile if its min and max x-coordinates are below this limit.
            Should not be larger than half the profile length.
        threshold (int, optional): Distance in millimeters from the centerline considered part of the profile.
            Should not exceed half the profile distance (max: 1000). Defaults to 700.
        plot (bool, optional): If True, generates a plot of the profiles. Defaults to True.

    Returns:
        list[pd.DataFrame]: A list of DataFrames, each representing an individual profile.
    """
    current_y = df.y.min()
    current_idx = 0

    diffs = []
    indices = []
    for idx, row in df.iterrows():
        if current_y - threshold <= row.y <= current_y + threshold:
            pass

        else:
            previous_y = df.loc[idx - 1].y
            diff = np.abs(row.y - previous_y)
            if diff > threshold:
                indices.append((current_idx, idx - 1))
                current_idx = idx
                current_y = row.y
                diffs.append(diff)

    if plot:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.hist(diffs, bins=100)

    profiles = []
    for idx in indices:
        profiles.append(df.iloc[idx[0] : idx[1]])

    return _filter_profiles(profiles, xlim=xlim)


def _filter_profiles(profiles: list[pd.DataFrame], xlim: int) -> list[pd.DataFrame]:
    """Filter profiles based on the minimum x-coordinate threshold. Drop profiles with min x-coordinate below the threshold.

    Parameters
    ----------
    profiles : list[pd.DataFrame]
        List of DataFrames representing profiles
    xlim : int
        Minimum x-coordinate threshold

    Returns
    -------
    list[pd.DataFrame]
        Filtered list of DataFrames representing profiles
    """

    return [profile for profile in profiles if np.abs(profile.x.min()) > xlim]


def get_xlim_for_filename(filename: str) -> int:
    """Return the minimum x-coordinate threshold for a given filename.

    Parameters
    ----------
    filename : str
        Filename of the CSV file

    Returns
    -------
    int
        Minimum x-coordinate threshold

    Raises
    ------
    NameError
        If the filename does not match any filter
    """
    filename = filename.split(".")[0]
    if "PtOut_BoH" in filename:
        if filename[-1] == "2":
            return 2500
        elif filename[-1] == "H":
            return 22000

    elif "ML_total" in filename:
        return 6000

    elif "AZ CrRo PoPr_red_vägyta" in filename:
        return 6000

    elif "Slope" in filename:
        if filename[-1] == "8":
            return 2500
        elif filename[-2:] == "12":
            return 1500
        elif filename[-2:] == "20":
            return 3000

    else:
        raise NameError("No filter for this file. Specify an ID filter")


def fit_poly_line(
    x: Sequence[float | int],
    y: Sequence[float | int],
    degree: int = 1,
) -> np.poly1d:
    """Fit a polynomial line to the given data points.

    Parameters
    ----------
    x : Sequence[Union[float, int]]
        x-coordinates of the data points
    y : Sequence[Union[float, int]]
        y-coordinates of the data points
    degree : int, optional
        Polynomial degree, by default 1

    Returns
    -------
    np.poly1d
        Polynomial line
    """
    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)
    return poly


def calc_rmse(residual: Sequence[float | int]) -> float:
    """Calculate the root mean square error of the residuals.

    Parameters
    ----------
    residual : Sequence[float  |  int]
        Residuals of the data points

    Returns
    -------
    float
        Root mean square error
    """
    if not isinstance(residual, np.ndarray):
        residual = np.array(residual)
    return np.sqrt(np.mean(residual**2))


def calc_energy(arr: Sequence[float | int]) -> float:
    """Calculate the energy of the given array.

    Parameters
    ----------
    arr : Sequence[float  |  int]
        Array of values

    Returns
    -------
    float
        Energy of the array
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return np.sum(arr**2)


def calc_energy_removed(y, residual):
    return (calc_energy(residual) / calc_energy(y)) * 100


def calc_residual(x, y):
    return x - y


def gen_path_img(filename: str) -> str:
    """Generate a path for saving images based on the filename.

    Parameters
    ----------
    filename : str
        Filename of the CSV file

    Returns
    -------
    str
        Path for saving images
    """
    global path_img_root
    path_img = os.path.join(path_img_root, filename.split(".")[0])
    os.makedirs(path_img, exist_ok=True)
    return path_img


def calc_slope(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> np.ndarray:
    """
    Iteratively calculate the slope between each pair of points in x and y.
    It follows the "the rise divided by the run" formula.

    :Example:
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 1, 2])
    >>> calc_slope(x, y)
    [1.0, 1.0]

    :param x: A numpy array or list of floats representing the x-coordinates of the points.
    :type x: Union[np.ndarray, list(float)]
    :param y: A numpy array or list of floats representing the y-coordinates of the points.
    :type y: Union[np.ndarray, list(float)]
    :return: A numpy array representing the slopes between each pair of points.
    :rtype: np.ndarray
    """
    if not isinstance(x, (np.ndarray, list)) or not isinstance(y, (np.ndarray, list)):
        raise TypeError("x and y must be numpy arrays or lists of floats")

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) < 2 or len(y) < 2:
        raise ValueError("x and y must have at least 2 elements")

    try:
        slopes = []
        for i in range(len(x) - 1):
            rise = y[i + 1] - y[i]
            run = x[i + 1] - x[i] + 0.01
            slopes.append(rise / run)

        return np.asarray(slopes)

    except Exception as e:
        raise ValueError(f"Error calculating slope: {e}")


def calc_slope_percentage(
    x: np.ndarray | list[float], y: np.ndarray | list[float]
) -> np.ndarray:
    """
    Iteratively calculate the slope in percentage between each pair of points in x and y.

    :Example:
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 1, 2])
    >>> calc_slope_percentage(x, y)
    [100.0, 100.0]

    :param x: A numpy array or list of floats representing the x-coordinates of the points.
    :type x: Union[np.ndarray, list(float)]
    :param y: A numpy array or list of floats representing the y-coordinates of the points.
    :type y: Union[np.ndarray, list(float)]
    :return: A numpy array representing the slopes in percentage between each pair of points.
    :rtype: np.ndarray
    """
    slopes: np.ndarray = calc_slope(x, y)
    return np.array([slope * 100 for slope in slopes])


def find_nearest_index(array, value):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def hide_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_histogram(df, col, xlabel, save_path=None):
    values = df[col].values
    values = values[~np.isnan(values)]

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(values, bins=30, kde=False, color="grey", edgecolor="black")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Antal", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    hide_spines(ax)

    ax.tick_params(axis="x", labelsize=14)  # X-axis tick labels font size
    ax.tick_params(axis="y", labelsize=14)  # Y-axis tick labels font size

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def split_into_row_chunks(data, chunk_size):
    chunks = []
    rows = data.shape[0]
    for i in range(0, rows, chunk_size):
        chunk = data.iloc[i : i + chunk_size, :]
        chunks.append(chunk)
    return chunks
