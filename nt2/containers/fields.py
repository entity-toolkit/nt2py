import h5py
import xarray as xr
import numpy as np
from dask.array.core import from_array
from dask.array.core import stack

from nt2.containers.container import Container
from nt2.containers.utils import _read_category_metadata_SingleFile


def _read_coordinates_SingleFile(coords: list[str], file: h5py.File):
    for st in file:
        group = file[st]
        if isinstance(group, h5py.Group):
            if any([k.startswith("X") for k in group if k is not None]):
                # cell-centered coords
                xc = {
                    c: (
                        np.asarray(xi[:])
                        if isinstance(xi := group[f"X{i+1}"], h5py.Dataset) and xi
                        else None
                    )
                    for i, c in enumerate(coords[::-1])
                }
                # cell edges
                xe_min = {
                    f"{c}_1": (
                        c,
                        (
                            np.asarray(xi[:-1])
                            if isinstance((xi := group[f"X{i+1}e"]), h5py.Dataset)
                            else None
                        ),
                    )
                    for i, c in enumerate(coords[::-1])
                }
                xe_max = {
                    f"{c}_2": (
                        c,
                        (
                            np.asarray(xi[1:])
                            if isinstance((xi := group[f"X{i+1}e"]), h5py.Dataset)
                            else None
                        ),
                    )
                    for i, c in enumerate(coords[::-1])
                }
                return {"x_c": xc, "x_emin": xe_min, "x_emax": xe_max}
        else:
            raise ValueError(f"Unexpected type {type(file[st])}")
    raise ValueError("Could not find coordinates in file")


def _preload_field_SingleFile(
    k: str,
    dim: int,
    ngh: int,
    outsteps: list[int],
    times: list[float],
    steps: list[int],
    coords: list[str],
    xc_coords: dict[str, str],
    xe_min_coords: dict[str, str],
    xe_max_coords: dict[str, str],
    coord_replacements: list[tuple[str, str]],
    field_replacements: list[tuple[str, str]],
    layout: str,
    file: h5py.File,
):
    if dim == 1:
        noghosts = slice(ngh, -ngh) if ngh > 0 else slice(None)
    elif dim == 2:
        noghosts = (slice(ngh, -ngh), slice(ngh, -ngh)) if ngh > 0 else slice(None)
    elif dim == 3:
        noghosts = (
            (slice(ngh, -ngh), slice(ngh, -ngh), slice(ngh, -ngh))
            if ngh > 0
            else slice(None)
        )
    else:
        raise ValueError("Invalid dimension")

    dask_arrays = []
    for s in outsteps:
        dset = file[f"{s}/{k}"]
        if isinstance(dset, h5py.Dataset):
            array = from_array(np.transpose(dset) if layout == "right" else dset)
            dask_arrays.append(array[noghosts])
        else:
            raise ValueError(f"Unexpected type {type(dset)}")

    k_ = k[1:]
    for c in coord_replacements:
        if "_" not in k_:
            k_ = k_.replace(c[0], c[1])
        else:
            k_ = "_".join([k_.split("_")[0].replace(c[0], c[1])] + k_.split("_")[1:])
    for f in field_replacements:
        k_ = k_.replace(*f)

    return k_, xr.DataArray(
        stack(dask_arrays, axis=0),
        dims=["t", *coords],
        name=k_,
        coords={
            "t": times,
            "s": ("t", steps),
            **xc_coords,
            **xe_min_coords,
            **xe_max_coords,
        },
    )


class FieldsContainer(Container):
    def __init__(self, **kwargs):
        super(FieldsContainer, self).__init__(**kwargs)
        QuantityDict = {
            "Ttt": "E",
            "Ttx": "Px",
            "Tty": "Py",
            "Ttz": "Pz",
        }
        CoordinateDict = {
            "cart": {"x": "x", "y": "y", "z": "z", "1": "x", "2": "y", "3": "z"},
            "sph": {
                "r": "r",
                "theta": "θ" if self.configs["use_greek"] else "th",
                "phi": "φ" if self.configs["use_greek"] else "ph",
                "1": "r",
                "2": "θ" if self.configs["use_greek"] else "th",
                "3": "φ" if self.configs["use_greek"] else "ph",
            },
        }
        if self.configs["single_file"]:
            assert self.master_file is not None, "Master file not found"
            self.metadata["fields"] = _read_category_metadata_SingleFile(
                "f", self.master_file
            )
        else:
            try:
                raise NotImplementedError("Multiple files not yet supported")
            except OSError:
                raise OSError(f"Could not open file {self.path}")

        coords = list(CoordinateDict[self.configs["coordinates"]].values())[::-1][
            -self.configs["dimension"] :
        ]

        if self.configs["single_file"]:
            self.mesh = _read_coordinates_SingleFile(coords, self.master_file)
        else:
            raise NotImplementedError("Multiple files not yet supported")

        self.fields = xr.Dataset()

        if len(self.metadata["fields"]["outsteps"]) > 0:
            if self.configs["single_file"]:
                for k in self.metadata["fields"]["quantities"]:
                    name, dset = _preload_field_SingleFile(
                        k,
                        dim=self.configs["dimension"],
                        ngh=self.configs["ngh"],
                        outsteps=self.metadata["fields"]["outsteps"],
                        times=self.metadata["fields"]["times"],
                        steps=self.metadata["fields"]["steps"],
                        coords=coords,
                        xc_coords=self.mesh["x_c"],
                        xe_min_coords=self.mesh["x_emin"],
                        xe_max_coords=self.mesh["x_emax"],
                        coord_replacements=list(
                            CoordinateDict[self.configs["coordinates"]].items()
                        ),
                        field_replacements=list(QuantityDict.items()),
                        layout=self.configs["layout"],
                        file=self.master_file,
                    )
                    self.fields[name] = dset
            else:
                raise NotImplementedError("Multiple files not yet supported")

    def __del__(self):
        if self.configs["single_file"] and self.master_file is not None:
            self.master_file.close()
        else:
            raise NotImplementedError("Multiple files not yet supported")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.configs["single_file"] and self.master_file is not None:
            self.master_file.close()
        else:
            raise NotImplementedError("Multiple files not yet supported")

    def print_fields(self) -> str:
        def sizeof_fmt(num, suffix="B"):
            for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
                if abs(num) < 1e3:
                    return f"{num:3.1f} {unit}{suffix}"
                num /= 1e3
            return f"{num:.1f} Y{suffix}"

        def compactify(lst):
            c = ""
            cntr = 0
            for l_ in lst:
                if cntr > 5:
                    c += "\n                "
                    cntr = 0
                c += l_ + ", "
                cntr += 1
            return c[:-2]

        string = ""
        field_keys = list(self.fields.data_vars.keys())

        if len(field_keys) > 0:
            string += "Fields:\n"
            string += f"  - data axes: {compactify(self.fields.indexes.keys())}\n"
            string += f"  - timesteps: {self.fields[field_keys[0]].shape[0]}\n"
            string += f"  - shape: {self.fields[field_keys[0]].shape[1:]}\n"
            string += f"  - quantities: {compactify(self.fields.data_vars.keys())}\n"
            string += f"  - total size: {sizeof_fmt(self.fields.nbytes)}\n"
        else:
            string += "Fields: empty\n"

        return string
