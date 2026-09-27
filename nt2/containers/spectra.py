from __future__ import annotations

from typing import Any

import dask
import dask.array as da
import numpy as np
import xarray as xr
from tqdm import tqdm

from ..utils import CoordinateSystem
from .base import BaseContainer


def remap_coords_cart(name: str) -> str:
    return {
        "X1": "x",
        "X2": "y",
        "X3": "z",
    }.get(name, name)


def remap_coords_sph(name: str) -> str:
    return {
        "X1": "r",
        "X2": "th",
        "X3": "ph",
    }.get(name, name)


class SpectraContainer(BaseContainer):
    """Parent class to manage the spectra dataframe."""

    __spectra_defined: bool = False
    __spectra: xr.Dataset | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(category="spectra", **kwargs)

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
    def spectra(self) -> xr.Dataset | None:
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
        spectra_names = {s for s in sorted(spectra_names) if s.startswith("sN")}
        ebin_name = "sEbn"
        first_spectrum_name = next(iter(spectra_names))
        shape = self.reader.ReadArrayShapeExplicitlyAtTimestep(
            self.path, "spectra", first_spectrum_name, first_step
        )

        energy_binedges = self.reader.ReadArrayAtTimestep(
            self.path, "spectra", ebin_name, first_step
        )
        num_spatial_dims = len(shape) - 1

        x_binedges = [
            self.reader.ReadArrayAtTimestep(
                self.path, "spectra", f"sX{i + 1}bn", first_step
            )
            for i in range(num_spatial_dims)
        ]

        def edges_to_bins(edges: np.ndarray) -> np.ndarray:
            diffs = np.diff(edges)
            if len(diffs) == 1 or np.isclose(
                diffs[1] - diffs[0], diffs[-1] - diffs[-2], atol=1e-2
            ):
                return 0.5 * (edges[1:] + edges[:-1])
            else:
                return (edges[1:] * edges[:-1]) ** 0.5

        attributes = self.reader.ReadAttrsAtTimestep(
            path=self.path, category="spectra", step=first_step
        )
        if self.coordinate_system is None:
            if "Coordinates" not in attributes:
                raise ValueError("Coordinates not found in attributes for particles.")
            if attributes["Coordinates"] in [b"cart", "cart"]:
                self.set_coordinate_system(CoordinateSystem.XYZ)
            elif attributes["Coordinates"] in [b"sph", "sph", b"qsph", "qsph"]:
                self.set_coordinate_system(CoordinateSystem.SPH)
            else:
                raise NotImplementedError(
                    f"Coordinate system {attributes['Coordinates']} not supported."
                )

        if self.remap is None or self.remap.get("coords", None) is None:
            self.set_remap(
                {
                    "coords": (
                        remap_coords_cart
                        if self.coordinate_system == CoordinateSystem.XYZ
                        else remap_coords_sph
                    ),
                }
            )

        ebins = edges_to_bins(energy_binedges)
        xbins = [edges_to_bins(xb) for xb in x_binedges]

        if self.remap is not None and "coords" in self.remap:
            new_xbins = {}
            for i in range(num_spatial_dims):
                new_xbins[self.remap["coords"](f"X{i + 1}")] = xbins[i]
            xbins = new_xbins
        else:
            xbins = {f"X{i + 1}": xbins[i] for i in range(num_spatial_dims)}

        all_dims = {
            "t": self.times,
            **xbins,
            "E": ebins,
        }
        all_coords = {**all_dims, "s": ("t", self.steps)}

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
                    spectra_names,
                    desc="spectra",
                    position=0,
                    leave=False,
                )
            },
            attrs=attributes,
        )

    @property
    def attrs(self) -> dict[str, Any]:
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
