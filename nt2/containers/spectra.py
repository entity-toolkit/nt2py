import os
import h5py
import xarray as xr

from nt2.containers.container import Container
from nt2.containers.utils import (
    _read_category_metadata,
    _read_spectra_species,
    _read_spectra_bins,
    _preload_spectra,
)


class SpectraContainer(Container):
    """
    * * * * SpectraContainer : Container * * * *

    Class for holding the spectra (energy distribution) data.

    Attributes
    ----------
    spectra : xarray.Dataset
        The xarray dataset of particle distributions.

    spectra_files : list
        The list of opened spectra files.

    Methods
    -------
    print_spectra()
        Prints the basic information about the spectra data.

    """

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
            self.metadata["spectra"] = _read_category_metadata(
                True, "s", self.master_file
            )
        else:
            spectra_path = os.path.join(self.path, "spectra")
            if os.path.isdir(spectra_path):
                files = sorted(os.listdir(spectra_path))
                try:
                    self.spectra_files = [
                        h5py.File(os.path.join(spectra_path, f), "r") for f in files
                    ]
                except OSError:
                    raise OSError(f"Could not open file {spectra_path}")
                self.metadata["spectra"] = _read_category_metadata(
                    False, "s", self.spectra_files
                )
        self._spectra = xr.Dataset()
        log_bins = self.attrs["output.spectra.log_bins"]

        if "spectra" in self.metadata and len(self.metadata["spectra"]["outsteps"]) > 0:
            if self.configs["single_file"]:
                assert self.master_file is not None, "Master file not found"
                species = _read_spectra_species(
                    f'Step{self.metadata["spectra"]["outsteps"][0]}', self.master_file
                )
                e_bins = _read_spectra_bins(
                    f'Step{self.metadata["spectra"]["outsteps"][0]}',
                    log_bins,
                    self.master_file,
                )
            else:
                species = _read_spectra_species("Step0", self.spectra_files[0])
                e_bins = _read_spectra_bins("Step0", log_bins, self.spectra_files[0])

            self.metadata["spectra"]["species"] = species

            for sp in species:
                self._spectra[f"n_{sp}"] = _preload_spectra(
                    self.configs["single_file"],
                    sp,
                    e_bins=e_bins,
                    outsteps=self.metadata["spectra"]["outsteps"],
                    times=self.metadata["spectra"]["times"],
                    steps=self.metadata["spectra"]["steps"],
                    file=(
                        self.master_file
                        if self.configs["single_file"] and self.master_file is not None
                        else self.spectra_files
                    ),
                )

    def __del__(self):
        if not self.configs["single_file"]:
            for f in self.spectra_files:
                f.close()

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
