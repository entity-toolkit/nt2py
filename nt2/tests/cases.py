import numpy as np
import nt2.readers.hdf5 as hdf5
import nt2.readers.adios2 as adios2

cwd = __file__.rsplit("/", 1)[0]

PARAMS: dict[str, dict[str, float | int]] = {
    "2D": {
        "sx1": 10,
        "sx2": 20,
        "nx1": 64,
        "nx2": 128,
    },
    "3D": {
        "sx1": 10,
        "sx2": 16,
        "sx3": 12.5,
        "nx1": 20,
        "nx2": 32,
        "nx3": 25,
    },
}

PARAMS["2D"]["dx"] = PARAMS["2D"]["sx1"] / PARAMS["2D"]["nx1"]
PARAMS["2D"]["x1min"] = -PARAMS["2D"]["sx1"] / 2 + PARAMS["2D"]["dx"] / 2
PARAMS["2D"]["x2min"] = -PARAMS["2D"]["sx2"] / 2 + PARAMS["2D"]["dx"] / 2
PARAMS["2D"]["dt"] = PARAMS["2D"]["dx"] / np.sqrt(2) / 2

PARAMS["3D"]["dx"] = PARAMS["3D"]["sx1"] / PARAMS["3D"]["nx1"]
PARAMS["3D"]["x1min"] = -PARAMS["3D"]["sx1"] / 2 + PARAMS["3D"]["dx"] / 2
PARAMS["3D"]["x2min"] = -PARAMS["3D"]["sx2"] / 2 + PARAMS["3D"]["dx"] / 2
PARAMS["3D"]["x3min"] = PARAMS["3D"]["dx"] / 2
PARAMS["3D"]["dt"] = PARAMS["3D"]["dx"] / np.sqrt(3) / 2

TESTS = [
    {
        "dim": "2D",
        "reader": hdf5.Reader,
        "path": f"{cwd}/testdata/h5_2d_cart_cpu/",
        "invalid_tstep": 71,
        "fields": True,
        "particles": True,
    },
    {
        "dim": "2D",
        "reader": hdf5.Reader,
        "path": f"{cwd}/testdata/h5_2d_cart_gpu/",
        "invalid_tstep": 81,
        "fields": True,
        "particles": True,
    },
    {
        "dim": "3D",
        "reader": hdf5.Reader,
        "path": f"{cwd}/testdata/h5_3d_cart_cpu/",
        "invalid_tstep": None,
        "fields": True,
        "particles": True,
    },
    {
        "dim": "3D",
        "reader": hdf5.Reader,
        "path": f"{cwd}/testdata/h5_3d_cart_gpu/",
        "invalid_tstep": None,
        "fields": False,
        "particles": True,
    },
    {
        "dim": "2D",
        "reader": adios2.Reader,
        "path": f"{cwd}/testdata/adios2_2d_cart_cpu/",
        "invalid_tstep": None,
        "fields": True,
        "particles": True,
    },
    {
        "dim": "3D",
        "reader": adios2.Reader,
        "path": f"{cwd}/testdata/adios2_3d_cart_cpu/",
        "invalid_tstep": None,
        "fields": True,
        "particles": True,
    },
]
