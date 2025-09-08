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
        "dt": 0.1443375647,
        "fields": {},
        "particles": {
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
            "num": 637,
        },
    },
    {
        "dim": "2D",
        "coords": "qsph",
        "reader": hdf5.Reader,
        "path": f"{cwd}/testdata/h5_2d_qsph_cpu/",
        "invalid_tstep": None,
        "dt": 2.93069752e-03,
        "fields": {
            "sx1": 1,
            "sx2": 30,
            "nx1": 512,
            "nx2": 256,
            "quantities": [
                "E1",
                "E2",
                "E3",
                "B1",
                "B2",
                "B3",
                "N_1",
                "N_2",
                "T00",
            ],
        },
        "particles": {
            "nspec": 2,
            "num": [2939, 1222],
        },
    },
    {
        "dim": "2D",
        "coords": "sph",
        "reader": adios2.Reader,
        "path": f"{cwd}/testdata/adios2_2d_sph_cpu/",
        "invalid_tstep": None,
        "dt": 5.99678652e-03,
        "fields": {
            "sx1": 1,
            "sx2": 30,
            "nx1": 512,
            "nx2": 256,
            "quantities": [
                "E1",
                "E2",
                "E3",
                "B1",
                "B2",
                "B3",
                "N_1",
                "N_2",
                "T00",
            ],
        },
        "particles": {
            "nspec": 2,
            "num": [253, 157],
        },
    },
]
