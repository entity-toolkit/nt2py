from typing import Any, Union, Dict

import dask
import dask.array as da
import xarray as xr
import numpy as np
from tqdm import tqdm

from .base import BaseContainer


class SpectraContainer(BaseContainer):
    """Parent class to manage the spectra dataframe."""

    __spectra_defined: bool = False
    __spectra: Union[xr.Dataset, None] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.valid_files = self.reader.GetValidFiles(
            path=self.path,
            category="spectra",
            num_cpus=self.num_cpus,
        )
        self.valid_steps = self.reader.GetValidSteps(
            path=self.path,
            category="spectra",
            num_cpus=self.num_cpus,
        )

        if self.reader.DefinesCategory(
            self.path,
            "spectra",
            self.valid_files,
        ):
            self.__spectra_defined = True
            self.__spectra = self._read_spectra()

    @property
    def spectra_defined(self) -> bool:
        """bool: Whether the spectra category is defined."""
        return self.__spectra_defined

    @property
    def spectra(self) -> Union[xr.Dataset, None]:
        """xr.Dataset: The spectra dataframe."""
        return self.__spectra

    def _read_spectrum(self, spectrum: str, step: int) -> Any:
        """Reads a spectrum from the data.

        This is a dask-delayed function used further to build the dataset.

        Parameters
        ----------
        spectrum : str
            Spectrum array to read.
        step : int
            Step to read.

        Returns
        -------
        Any
            Spectrum data.

        """
        return self.reader.ReadArrayAtTimestep(self.path, "spectra", spectrum, step)

    def _read_spectra(self) -> xr.Dataset:
        if self.verify:
            self.reader.VerifySameCategoryNames(
                self.path,
                "spectra",
                "s",
                self.valid_steps,
            )
        first_step = self.valid_steps[0]
        spectra_names = self.reader.ReadCategoryNamesAtTimestep(
            self.path, "spectra", "s", first_step
        )
        spectra_names = set(s for s in sorted(spectra_names) if s.startswith("sN"))
        ebin_name = "sEbn"
        first_spectrum_name = next(iter(spectra_names))
        shape = self.reader.ReadArrayShapeExplicitlyAtTimestep(
            self.path, "spectra", first_spectrum_name, first_step
        )
        times = self.reader.ReadPerTimestepVariable(
            self.path,
            "spectra",
            "Time",
            "t",
            self.valid_files,
        )
        steps = self.reader.ReadPerTimestepVariable(
            self.path,
            "spectra",
            "Step",
            "s",
            self.valid_files,
        )

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

        attributes = self.reader.ReadAttrsAtTimestep(
            path=self.path, category="spectra", step=first_step
        )

        def remap_name(name: str) -> str:
            return name[1:]

        return xr.Dataset(
            {
                remap_name(spectrum): xr.DataArray(
                    da.stack(
                        [
                            da.from_delayed(
                                dask.delayed(self._read_spectrum)(
                                    spectrum=spectrum,
                                    step=step,
                                ),
                                shape=shape,
                                dtype="float",
                            )
                            for step in tqdm(
                                self.valid_steps,
                                desc="steps",
                                position=1,
                                leave=False,
                            )
                        ],
                    ),
                    name=remap_name(spectrum),
                    dims=all_dims,
                    coords=all_coords,
                )
                for spectrum in tqdm(
                    spectra_names, desc="spectra", position=0, leave=False
                )
            },
            attrs=attributes,
        )

    @property
    def attrs(self) -> Dict[str, Any]:
        """dict: The attributes of the spectra dataframe."""
        if self.spectra_defined:
            return self.spectra.attrs
        else:
            return {}

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
