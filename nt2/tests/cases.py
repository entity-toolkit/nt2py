import nt2.readers.hdf5 as hdf5
import nt2.readers.adios2 as adios2

cwd = __file__.rsplit("/", 1)[0]

TESTS = [
    {
        "dim": "2D",
        "reader": hdf5.Reader,
        "path": f"{cwd}/testdata/h5_2d_cart_cpu/",
        "invalid_tstep": 71,
        "fields": {
            "sx1": 10,
            "sx2": 20,
            "nx1": 64,
            "nx2": 128,
        },
        "particles": {
            "num": [325, 325, 324, 325],
        },
    },
    {
        "dim": "2D",
        "reader": hdf5.Reader,
        "path": f"{cwd}/testdata/h5_2d_cart_gpu/",
        "invalid_tstep": 81,
        "fields": {
            "sx1": 10,
            "sx2": 20,
            "nx1": 64,
            "nx2": 128,
        },
        "particles": {
            "dt": None,
            "num": 327,
        },
    },
    {
        "dim": "3D",
        "reader": hdf5.Reader,
        "path": f"{cwd}/testdata/h5_3d_cart_cpu/",
        "invalid_tstep": None,
        "fields": {
            "sx1": 10,
            "sx2": 16,
            "sx3": 12.5,
            "nx1": 20,
            "nx2": 32,
            "nx3": 25,
        },
        "particles": {
            "num": 639,
        },
    },
    {
        "dim": "3D",
        "reader": hdf5.Reader,
        "path": f"{cwd}/testdata/h5_3d_cart_gpu/",
        "invalid_tstep": None,
        "fields": {},
        "particles": {
            "dt": 0.1443375647,
            "num": 640,
        },
    },
    {
        "dim": "2D",
        "reader": adios2.Reader,
        "path": f"{cwd}/testdata/adios2_2d_cart_cpu/",
        "invalid_tstep": None,
        "fields": {
            "sx1": 10,
            "sx2": 20,
            "nx1": 64,
            "nx2": 128,
        },
        "particles": {
            "dt": None,
            "num": [323, 324, 323, 324],
        },
    },
    {
        "dim": "3D",
        "reader": adios2.Reader,
        "path": f"{cwd}/testdata/adios2_3d_cart_cpu/",
        "invalid_tstep": None,
        "fields": {
            "sx1": 10,
            "sx2": 16,
            "sx3": 12.5,
            "nx1": 20,
            "nx2": 32,
            "nx3": 25,
        },
        "particles": {
            "dt": None,
            "num": 637,
        },
    },
]
