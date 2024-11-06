import h5py
import numpy as np
import xarray as xr
from dask.array.core import from_array
from dask.array.core import stack

from nt2.containers.container import Container
from nt2.containers.utils import _read_category_metadata_SingleFile


def _read_species_SingleFile(first_step: int, file: h5py.File):
    group = file[first_step]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Unexpected type {type(group)}")
    species = np.unique(
        [int(pq.split("_")[1]) for pq in group.keys() if pq.startswith("sN")]
    )
    return species


def _read_spectra_bins_SingleFile(first_step: int, log_bins: bool, file: h5py.File):
    group = file[first_step]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Unexpected type {type(group)}")
    e_bins = group["sEbn"]
    if not isinstance(e_bins, h5py.Dataset):
        raise ValueError(f"Unexpected type {type(e_bins)}")
    if log_bins:
        e_bins = np.sqrt(e_bins[1:] * e_bins[:-1])
    else:
        e_bins = (e_bins[1:] + e_bins[:-1]) / 2
    return e_bins


def _preload_spectra_SingleFile(
    sp: int,
    e_bins: np.ndarray,
    outsteps: list[int],
    times: list[float],
    steps: list[int],
    file: h5py.File,
):
    dask_arrays = []
    for st in outsteps:
        array = from_array(file[f"{st}/sN_{sp}"])
        dask_arrays.append(array)

    return xr.DataArray(
        stack(dask_arrays, axis=0),
        dims=["t", "e"],
        name=f"n_{sp}",
        coords={
            "t": times,
            "s": ("t", steps),
            "e": e_bins,
        },
    )


class SpectraContainer(Container):
    def __init__(self, **kwargs):
        super(SpectraContainer, self).__init__(**kwargs)
        assert "single_file" in self.configs
        assert "use_pickle" in self.configs
        assert "use_greek" in self.configs
        assert "path" in self.__dict__
        assert "metadata" in self.__dict__
        assert "mesh" in self.__dict__
        assert "attrs" in self.__dict__

        if self.configs["single_file"]:
            assert self.master_file is not None, "Master file not found"
            self.metadata["spectra"] = _read_category_metadata_SingleFile(
                "s", self.master_file
            )
        self._spectra = xr.Dataset()
        log_bins = self.attrs["output.spectra.log_bins"]

        if len(self.metadata["spectra"]["outsteps"]) > 0:
            if self.configs["single_file"]:
                assert self.master_file is not None, "Master file not found"
                species = _read_species_SingleFile(
                    self.metadata["spectra"]["outsteps"][0], self.master_file
                )
            else:
                raise NotImplementedError("Multiple files not yet supported")

            e_bins = _read_spectra_bins_SingleFile(
                self.metadata["spectra"]["outsteps"][0], log_bins, self.master_file
            )

            for sp in species:
                self._spectra[f"n_{sp}"] = _preload_spectra_SingleFile(
                    sp,
                    e_bins,
                    self.metadata["spectra"]["outsteps"],
                    self.metadata["spectra"]["times"],
                    self.metadata["spectra"]["steps"],
                    self.master_file,
                )

    @property
    def spectra(self):
        return self._spectra

    def print_spectra(self) -> str:
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
        spec_keys = list(self.spectra.data_vars.keys())

        if len(spec_keys) > 0:
            string += "Spectra:\n"
            string += f"  - data axes: {compactify(self.spectra.indexes.keys())}\n"
            string += f"  - timesteps: {self.spectra[spec_keys[0]].shape[0]}\n"
            string += f"  - # of bins: {self.spectra[spec_keys[0]].shape[1]}\n"
            string += f"  - quantities: {compactify(self.spectra.data_vars.keys())}\n"
            string += f"  - total size: {sizeof_fmt(self.spectra.nbytes)}\n"
        else:
            string += "Spectra: empty\n"

        return string
