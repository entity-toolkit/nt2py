from enum import Enum
import os
import re
import inspect

import xarray as xr


class FutureDeprecationWarning(Warning):
    pass


class Format(Enum):
    HDF5 = "h5"
    BP5 = "bp"


class Layout(Enum):
    L = "Left"
    R = "Right"


class CoordinateSystem(Enum):
    XYZ = "Cartesian"
    SPH = "Spherical"


def DetermineDataFormat(path: str) -> Format:
    """Determine the data format for the files in the given path.

    Parameters
    ----------
    path : str

    Returns
    -------
    Format
        The data format of the file.

    Raises
    -------
    ValueError
        If the file format is unknown.
    """
    categories = ["fields", "particles", "spectra"]
    for category in categories:
        category_path = os.path.join(path, category)
        if os.path.exists(category_path):
            files = [
                f
                for f in os.listdir(category_path)
                if re.match(rf"^{category}\.\d{{{8}}}\.", f)
            ]
            if len(files) > 0:
                filename = files[0]
                if filename.endswith(".h5"):
                    return Format.HDF5
                elif filename.endswith(".bp"):
                    return Format.BP5
                else:
                    raise ValueError(f"Unknown file format: {filename}.")
    raise ValueError("Could not determine file format.")


def ToHumanReadable(num: float | int, suffix="B") -> str:
    """Convert a number to a human-readable format with SI prefixes.

    Parameters
    ----------
    num : float | int
        The number to convert.
    suffix : str
        The suffix to append to the number (default: "B").

    Returns
    -------
    str
        The number in human-readable format with SI prefixes.
    """
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < 1e3:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1e3
    return f"{num:.1f} Y{suffix}"


def DataIs2DPolar(ds: xr.Dataset) -> bool:
    """Check if the dataset is 2D polar.
    A dataset is considered 2D polar if it has two dimensions: "r" and either "θ" or "th".

    Returns
    -------
    bool
        True if the dataset is 2D polar, False otherwise.
    """
    return ("r" in ds.dims and ("θ" in ds.dims or "th" in ds.dims)) and len(
        ds.dims
    ) == 2


def InheritClassDocstring(cls):
    """Decorator to inherit docstring from parent classes.
    This decorator appends the docstrings of all parent classes to the docstring of the class.
    """
    if cls.__doc__ is None:
        cls.__doc__ = ""
    for base in inspect.getmro(cls):
        if base.__doc__ is not None:
            cls.__doc__ += base.__doc__
    return cls
