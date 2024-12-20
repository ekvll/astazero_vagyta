import numpy as np


def check_array(arr):
    check_len(arr)
    check_nan(arr)
    constant_array(arr)
    flat_section(arr)
    return numpy_compatible(arr)


def check_len(arr):
    if len(arr) < 3:
        raise Exception("Array is too short")


def check_nan(arr):
    if np.any(np.isnan(arr)):
        raise Exception("Array contains NaN values")


def constant_array(arr):
    if np.all(arr == arr[0]):
        raise Exception("Array is constant")


def flat_section(arr):
    if np.all(np.diff(arr) == 0):
        raise Exception("Array is flat")


def numpy_compatible(arr):
    if not isinstance(arr, np.ndarray):
        try:
            return np.asarray(arr)
        except Exception:
            raise Exception("Array is not numpy compatible")
    return arr
