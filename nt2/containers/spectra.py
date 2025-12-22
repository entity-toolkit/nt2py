from typing import Any

import dask
import dask.array as da
import xarray as xr
import numpy as np

from nt2.containers.container import BaseContainer
from nt2.readers.base import BaseReader


class Spectra(BaseContainer):
    """Parent class to manager the spectra dataframe."""

    @staticmethod
    @dask.delayed
    def __read_spectrum(path: str, reader: BaseReader, spectrum: str, step: int) -> Any:
        """Reads a spectrum from the data.

        This is a dask-delayed function used further to build the dataset.

        Parameters
        ----------
        path : str
            Main path to the data.
        reader : BaseReader
            Reader to use to read the data.
        spectrum : str
            Spectrum array to read.
        step : int
            Step to read.

        Returns
        -------
        Any
            Spectrum data.

        """
        return reader.ReadArrayAtTimestep(path, "spectra", spectrum, step)

    def __init__(self, **kwargs: Any) -> None:
        super(Spectra, self).__init__(**kwargs)
        if self.reader.DefinesCategory(self.path, "spectra"):
            self.__spectra_defined = True
            self.__spectra = self.__read_spectra()
        else:
            self.__spectra_defined = False
            self.__spectra = xr.Dataset()

    @property
    def spectra_defined(self) -> bool:
        """bool: Whether the spectra category is defined."""
        return self.__spectra_defined

    @property
    def spectra(self) -> xr.Dataset:
        """xr.Dataset: The spectra dataframe."""
        return self.__spectra

    def __read_spectra(self) -> xr.Dataset:
        self.reader.VerifySameCategoryNames(self.path, "spectra", "s")
        valid_steps = sorted(self.reader.GetValidSteps(self.path, "spectra"))
        spectra_names = self.reader.ReadCategoryNamesAtTimestep(
            self.path, "spectra", "s", valid_steps[0]
        )
        spectra_names = set(s for s in sorted(spectra_names) if s.startswith("sN"))
        ebin_name = "sEbn"
        first_step = valid_steps[0]
        first_spectrum_name = next(iter(spectra_names))
        shape = self.reader.ReadArrayShapeExplicitlyAtTimestep(
            self.path, "spectra", first_spectrum_name, first_step
        )
        times = self.reader.ReadPerTimestepVariable(self.path, "spectra", "Time", "t")
        steps = self.reader.ReadPerTimestepVariable(self.path, "spectra", "Step", "s")

        ebins = self.reader.ReadArrayAtTimestep(
            self.path, "spectra", ebin_name, first_step
        )

        diffs = np.diff(ebins)
        if np.isclose(diffs[1] - diffs[0], diffs[-1] - diffs[-2], atol=1e-2):
            ebins = 0.5 * (ebins[1:] + ebins[:-1])
        else:
            ebins = (ebins[1:] * ebins[:-1]) ** 0.5

        all_dims = {**times, "E": ebins}
        all_coords = {**all_dims, "s": ("t", steps["s"])}

        def remap_name(name: str) -> str:
            return name[1:]

        return xr.Dataset(
            {
                remap_name(spectrum): xr.DataArray(
                    da.stack(
                        [
                            da.from_delayed(
                                self.__read_spectrum(
                                    path=self.path,
                                    reader=self.reader,
                                    spectrum=spectrum,
                                    step=step,
                                ),
                                shape=shape,
                                dtype="float",
                            )
                            for step in valid_steps
                        ],
                    ),
                    name=remap_name(spectrum),
                    dims=all_dims,
                    coords=all_coords,
                )
                for spectrum in spectra_names
            },
            attrs=self.reader.ReadAttrsAtTimestep(
                path=self.path, category="spectra", step=first_step
            ),
        )

    def help_spectra(self, prepend="") -> str:
        ret = f"{prepend}- use .sel(...) to select specific energy or time intervals\n"
        ret += f"{prepend}  t  : time (float)\n"
        ret += f"{prepend}  st : step (int)\n"
        ret += f"{prepend}  E  : energy bin (float)\n{prepend}\n"
        ret += f"{prepend}  # example:\n"
        ret += f"{prepend}  #   .sel(E=slice(10.0, 20.0)).sel(t=0, method='nearest')\n{prepend}\n"
        ret += f"{prepend}- use .isel(...) to select spectra based on energy bin or time index:\n"
        ret += f"{prepend}  t  : timestamp index (int)\n"
        ret += f"{prepend}  st : step index (int)\n"
        ret += f"{prepend}  E  : energy bin index (int)\n{prepend}\n"
        ret += f"{prepend}  # example:\n"
        ret += f"{prepend}  #   .isel(t=-1, E=11)\n"
        ret += f"{prepend}\n"
        ret += f"{prepend}  # example:\n"
        ret += f"{prepend}  #  .spectra.N_1.sel(E=slice(None, 50)).isel(t=5).plot()\n"
        return ret
