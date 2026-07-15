from typing import Callable, Any, Union, Optional, List, Dict, Tuple

import os

import xarray as xr
import pandas as pd

from ..utils import (
    ToHumanReadable,
    DetermineDataFormat,
    Format,
    CoordinateSystem,
    CoordinateSystemType,
)
from ..readers.base import BaseReader
from ..readers.hdf5 import Reader as HDF5Reader
from ..readers.adios2 import Reader as BP5Reader

from .fields import FieldContainer
from .particles import ParticleContainer
from .particle_dataset import ParticleDataset
from .spectra import SpectraContainer
from .diagnostics import Diagnostics

from ..plotters.export import makeFramesAndMovie


def compactify(lst: Union[List[Any], Any]) -> str:
    c = ""
    cntr = 0
    for l_ in lst:
        if cntr > 5:
            c += "\n|   "
            cntr = 0
        c += f"{l_}, "
        cntr += 1
    return c[:-2]


class Data:
    """Main class to manage all the data containers."""

    _fields: Optional[FieldContainer] = None
    _particles: Optional[ParticleContainer] = None
    _spectra: Optional[SpectraContainer] = None
    _diagnostics: Optional[Diagnostics] = None

    def __init__(
        self,
        path: str,
        fields: bool = True,
        particles: bool = True,
        spectra: bool = True,
        diagnostics: bool = False,
        verify: bool = False,
        timerange: Optional[Tuple[Union[float, None], Union[float, None]]] = None,
        steprange: Optional[Tuple[Union[int, None], Union[int, None]]] = None,
        reader: Optional[BaseReader] = None,
        remap: Optional[Dict[str, Callable[[str], str]]] = None,
        coord_system: Optional[CoordinateSystemType] = None,
        num_cpus: Optional[int] = min(os.cpu_count() or 1, 16),
    ):
        """Initializer for the Data class.

        Parameters
        ----------
        path : str
            Main path to the data
        components : list[OutputComponentType], optional
            List of components to load. If None, all components will be loaded.
        fields : bool, optional
            Whether to load the fields component. Default is True.
        particles : bool, optional
            Whether to load the particles component. Default is True.
        spectra : bool, optional
            Whether to load the spectra component. Default is True.
        diagnostics : bool, optional
            Whether to load the diagnostics component. Default is False.
        verify : bool, optional
            Whether to verify the data. Default is False.
        timerange : tuple[float | None, float | None], optional
            Time range to load. If None, all times will be loaded.
        steprange : tuple[int | None, int | None], optional
            Step range to load. If None, all steps will be loaded.
        reader : BaseReader, optional
            Reader to use to read the data. If None, it will be determined
            based on the file format.
        remap : dict[str, Callable[[str], str]], optional
            Remap dictionary to use to remap the data names (coords, fields, etc.).
        coord_system : Literal["XYZ", "SPH"], optional
            Coordinate system of the data. If None, it will be determined
            based on the data attrs (if remap is also None).
        num_cpus : int, optional
            Number of CPUs to use for parallel processing. If None, it will use all available CPUs.

        Raises
        ------
        NotImplementedError
            If the data format or coordinate system support is not implemented.
        ValueError
            If the reader format does not match the data format or if coordinate system cannot be inferred.
        """
        # determine the reader from the format
        fmt = DetermineDataFormat(path)
        if reader is None:
            if fmt == Format.HDF5:
                __reader = HDF5Reader()
            elif fmt == Format.BP5:
                __reader = BP5Reader()
            else:
                raise NotImplementedError(
                    "Only HDF5 & BP5 formats are supported at the moment."
                )
        else:
            if fmt != reader.format:
                raise ValueError(
                    f"Reader format {reader.format} does not match data format {fmt}."
                )
            __reader = reader

        if fields:
            self._fields = FieldContainer(
                path=path,
                reader=__reader,
                verify=verify,
                timerange=timerange,
                steprange=steprange,
                remap=remap,
                coord_system=CoordinateSystem.from_str(coord_system)
                if coord_system
                else None,
                num_cpus=num_cpus,
            )
        if particles:
            self._particles = ParticleContainer(
                path=path,
                reader=__reader,
                verify=verify,
                timerange=timerange,
                steprange=steprange,
                remap=remap,
                coord_system=CoordinateSystem.from_str(coord_system)
                if coord_system
                else None,
                num_cpus=num_cpus,
            )
        if spectra:
            self._spectra = SpectraContainer(
                path=path,
                reader=__reader,
                verify=verify,
                timerange=timerange,
                steprange=steprange,
                remap=remap,
                coord_system=CoordinateSystem.from_str(coord_system)
                if coord_system
                else None,
                num_cpus=num_cpus,
            )
        if diagnostics:
            self._diagnostics = Diagnostics(path=path)
        self.__attrs: Dict[str, Any] = {}
        if self.fields_defined:
            self.__attrs.update(**self._fields.attrs)
        if self.particles_defined:
            self.__attrs.update(**self._particles.attrs)
        if self.spectra_defined:
            self.__attrs.update(**self._spectra.attrs)

    @property
    def fields_defined(self) -> bool:
        """bool: Whether fields are defined in the data."""
        return (
            self._fields is not None
            and self._fields.fields_defined
            and self._fields.fields is not None
        )

    @property
    def particles_defined(self) -> bool:
        """bool: Whether particles are defined in the data."""
        return (
            self._particles is not None
            and self._particles.particles_defined
            and self._particles.particles is not None
        )

    @property
    def spectra_defined(self) -> bool:
        """bool: Whether spectra are defined in the data."""
        return (
            self._spectra is not None
            and self._spectra.spectra_defined
            and self._spectra.spectra is not None
        )

    @property
    def diagnostics_defined(self) -> bool:
        """bool: Whether diagnostics are defined in the data."""
        return self._diagnostics is not None and self._diagnostics.df is not None

    @property
    def fields(self) -> xr.Dataset:
        """xr.Dataset: The fields dataset."""
        if not self.fields_defined:
            raise ValueError("Fields are not defined in the data.")
        return self._fields.fields

    @property
    def particles(self) -> ParticleDataset:
        """ParticleDataset: The particles dataset."""
        if not self.particles_defined:
            raise ValueError("Particles are not defined in the data.")
        return self._particles.particles

    @property
    def spectra(self) -> xr.Dataset:
        """xr.Dataset: The spectra dataset."""
        if not self.spectra_defined:
            raise ValueError("Spectra are not defined in the data.")
        return self._spectra.spectra

    @property
    def diagnostics(self) -> pd.DataFrame:
        """Diagnostics: The diagnostics dataset."""
        if not self.diagnostics_defined:
            raise ValueError("Diagnostics are not defined in the data.")
        return self._diagnostics.df

    @property
    def attrs(self) -> Dict[str, Any]:
        """dict: The attributes of the data."""
        return self.__attrs

    def makeMovie(
        self,
        plot: Callable,
        time: Optional[List[float]] = None,
        num_cpus: Optional[int] = None,
        **movie_kwargs: Any,
    ) -> bool:
        """Create animation with provided plot function.

        Parameters
        ----------
        plot : callable
            A function that takes a single argument (time in physical units) and produces a plot.
        time : array_like, optional
            An array of time values to use for the animation. If not provided, the entire time range will be used.
        num_cpus : int, optional
            The number of CPUs to use for parallel processing. If None, it will use all available CPUs.
        **movie_kwargs : dict
            Additional keyword arguments to pass to the movie creation function.

        Returns
        -------
        bool
            True if the movie was created successfully, False otherwise.
        """
        if time is None:
            if self.fields_defined:
                time = self.fields.t.values
            elif self.particles_defined:
                time = list(self.particles.times)
            else:
                raise ValueError("No time values found.")
        if time is None:
            raise ValueError("No time values found.")
        name: str = ""
        provided_name = movie_kwargs.pop("name", None)
        if provided_name is not None:
            name = provided_name
        elif self.attrs.get("simulation.name", None) is None:
            name = provided_name
        else:
            name_b = self.attrs.get("simulation.name")
            if isinstance(name_b, bytes):
                name = name_b.decode("utf-8")
            else:
                name = str(name_b)
        if name is None:
            raise ValueError("No name provided for the movie.")
        return makeFramesAndMovie(
            name=name,
            data=self,
            plot=plot,
            times=time,
            num_cpus=num_cpus,
            **movie_kwargs,
        )

    def to_str(self) -> str:
        """str: String representation of the all the enclosed dataframes."""

        string = ""
        if self.fields_defined:
            string += "FieldsDataset:\n"
            string += "==============\n"
            string += f"| Coordinates:\n|   {self._fields.coordinate_system.value}\n|\n"
            string += f"| Data axes:\n|   {compactify(self.fields.indexes.keys())}\n|\n"
            delta_t = (
                self.fields.coords["t"].values[1] - self.fields.coords["t"].values[0]
            ) / (self.fields.coords["s"].values[1] - self.fields.coords["s"].values[0])
            string += f"|   - dt: {delta_t:.2e}\n"
            for key in self.fields.coords.keys():
                crd = self.fields.coords[key].values
                fmt = ""
                if key != "s":
                    fmt = ".2f"
                string += f"|   - {key}: {crd.min():{fmt}} -> {crd.max():{fmt}} [{len(crd)}]\n"
            string += "|\n"
            string += f"| Quantities:\n|   {compactify(sorted(map(str, self.fields.data_vars.keys())))}\n|\n"
            string += f"| Total size: {ToHumanReadable(self.fields.nbytes)}\n\n"
        else:
            string += "FieldsDataset:\n"
            string += "==============\n"
            string += "  empty\n\n"
        if self.particles_defined:
            species = sorted(self.particles.species)
            string += "ParticleDataset:\n"
            string += "================\n"
            string += f"| Species:\n|   {compactify(species)}\n|\n"
            string += f"| Timesteps:\n|   {len(self.particles.times)}\n|\n"
            string += f"| Quantities:\n|   {compactify(self.particles.columns)}\n|\n"
            string += (
                f"| Estimated index size: {ToHumanReadable(self.particles.nbytes)}\n|\n"
            )
            string += self._particles.help_particles("| ")
            string += "\n"
        else:
            string += "ParticleDataset:\n"
            string += "================\n"
            string += "  empty\n\n"
        if self.spectra_defined and self.spectra is not None:
            string += "SpectraDataset:\n"
            string += "===============\n"
            string += (
                f"| Data axes:\n|   {compactify(self.spectra.indexes.keys())}\n|\n"
            )
            delta_t = (
                self.spectra.coords["t"].values[1] - self.spectra.coords["t"].values[0]
            ) / (
                self.spectra.coords["s"].values[1] - self.spectra.coords["s"].values[0]
            )
            string += f"|   - dt: {delta_t:.2e}\n"
            for key in self.spectra.coords.keys():
                crd = self.spectra.coords[key].values
                fmt = ""
                if key != "s":
                    fmt = ".2f"
                string += f"|   - {key}: {crd.min():{fmt}} -> {crd.max():{fmt}} [{len(crd)}]\n"
            string += "|\n"
            string += f"| Quantities:\n|   {compactify(sorted(map(str, self.spectra.data_vars.keys())))}\n|\n"
            string += f"| Total size: {ToHumanReadable(self.spectra.nbytes)}\n|\n"
            string += self._spectra.help_spectra("| ")
        else:
            string += "SpectraDataset:\n"
            string += "===============\n"
            string += "  empty\n\n"

        return string

    def __str__(self) -> str:
        return self.to_str()

    def __repr__(self) -> str:
        return self.to_str()
