from typing import Callable, Any

import sys

if sys.version_info >= (3, 12):
    from typing import override
else:

    def override(method):
        return method


from collections.abc import KeysView
from nt2.utils import ToHumanReadable

import xarray as xr

from nt2.utils import (
    DetermineDataFormat,
    InheritClassDocstring,
    Format,
    CoordinateSystem,
)
from nt2.readers.base import BaseReader
from nt2.readers.hdf5 import Reader as HDF5Reader
from nt2.readers.adios2 import Reader as BP5Reader
from nt2.containers.fields import Fields
from nt2.containers.particles import Particles

import nt2.plotters.polar as acc_polar
import nt2.plotters.particles as acc_particles
import nt2.plotters.inspect as acc_inspect
import nt2.plotters.movie as acc_movie
from nt2.plotters.export import makeFramesAndMovie


@xr.register_dataset_accessor("polar")
@InheritClassDocstring
class DatasetPolarPlotAccessor(acc_polar.ds_accessor):
    pass


@xr.register_dataset_accessor("particles")
@InheritClassDocstring
class DatasetParticlesPlotAccessor(acc_particles.ds_accessor):
    pass


@xr.register_dataarray_accessor("polar")
@InheritClassDocstring
class PolarPlotAccessor(acc_polar.accessor):
    pass


@xr.register_dataset_accessor("inspect")
@InheritClassDocstring
class DatasetInspectPlotAccessor(acc_inspect.ds_accessor):
    pass


@xr.register_dataarray_accessor("movie")
@InheritClassDocstring
class MoviePlotAccessor(acc_movie.accessor):
    pass


class Data(Fields, Particles):
    """Main class to manage all the data containers.

    Inherits from all category-specific containers.

    """

    def __init__(
        self,
        path: str,
        reader: BaseReader | None = None,
        remap: dict[str, Callable[[str], str]] | None = None,
        coord_system: CoordinateSystem | None = None,
    ):
        """Initializer for the Data class.

        Parameters
        ----------
        path : str
            Main path to the data
        reader : BaseReader | None
            Reader to use to read the data. If None, it will be determined
            based on the file format.
        remap : dict[str, Callable[[str], str]] | None
            Remap dictionary to use to remap the data names (coords, fields, etc.).
        coord_system : CoordinateSystem | None
            Coordinate system of the data. If None, it will be determined
            based on the data attrs (if remap is also None).

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
                self.__reader = HDF5Reader()
            elif fmt == Format.BP5:
                self.__reader = BP5Reader()
            else:
                raise NotImplementedError(
                    "Only HDF5 & BP5 formats are supported at the moment."
                )
        else:
            if fmt != reader.format:
                raise ValueError(
                    f"Reader format {reader.format} does not match data format {fmt}."
                )
            self.__reader = reader

        # determine the coordinate system and remapping
        self.__attrs: dict[str, Any] = {}
        for category in ["fields", "particles", "spectra"]:
            if self.__reader.DefinesCategory(path, category):
                valid_steps = self.__reader.GetValidSteps(path, category)
                if len(valid_steps) == 0:
                    raise ValueError(f"No valid steps found for category {category}.")
                first_step = valid_steps[0]
                attrs = self.__reader.ReadAttrsAtTimestep(path, category, first_step)
                self.__attrs.update(**attrs)
                if "Coordinates" not in attrs:
                    raise ValueError(
                        f"Coordinates not found in attributes for category {category}."
                    )
                else:
                    if attrs["Coordinates"] in [b"cart", "cart"]:

                        def remap_fields(name: str) -> str:
                            name = name[1:]
                            fieldname = name.split("_")[0]
                            fieldname = fieldname.replace("0", "t")
                            fieldname = fieldname.replace("1", "x")
                            fieldname = fieldname.replace("2", "y")
                            fieldname = fieldname.replace("3", "z")
                            suffix = "_".join(name.split("_")[1:])
                            return f"{fieldname}{'_' + suffix if suffix != '' else ''}"

                        def remap_coords(name: str) -> str:
                            return {
                                "X1": "x",
                                "X2": "y",
                                "X3": "z",
                            }.get(name, name)

                        def remap_prtl_quantities(name: str) -> str:
                            shortname = name[1:]
                            return {
                                "X1": "x",
                                "X2": "y",
                                "X3": "z",
                                "U1": "ux",
                                "U2": "uy",
                                "U3": "uz",
                                "W": "w",
                            }.get(shortname, shortname)

                        coord_system = CoordinateSystem.XYZ

                    elif attrs["Coordinates"] in [b"sph", "sph", b"qsph", "qsph"]:

                        def remap_fields(name: str) -> str:
                            name = name[1:]
                            fieldname = name.split("_")[0]
                            fieldname = fieldname.replace("0", "t")
                            fieldname = fieldname.replace("1", "r")
                            fieldname = fieldname.replace("2", "th")
                            fieldname = fieldname.replace("3", "ph")
                            suffix = "_".join(name.split("_")[1:])
                            return f"{fieldname}{'_' + suffix if suffix != '' else ''}"

                        def remap_coords(name: str) -> str:
                            return {
                                "X1": "r",
                                "X2": "th",
                                "X3": "ph",
                            }.get(name, name)

                        def remap_prtl_quantities(name: str) -> str:
                            shortname = name[1:]
                            return {
                                "X1": "r",
                                "X2": "th",
                                "X3": "ph",
                                "U1": "ur",
                                "U2": "uth",
                                "U3": "uph",
                                "W": "w",
                            }.get(shortname, shortname)

                        coord_system = CoordinateSystem.SPH

                    else:
                        raise NotImplementedError(
                            f"Coordinate system {attrs['Coordinates']} not supported."
                        )
                    if remap is None:
                        remap = {
                            "coords": remap_coords,
                            "fields": remap_fields,
                            "particles": remap_prtl_quantities,
                        }
                    break

        if coord_system is None:
            raise ValueError("No coordinate system found in the data.")

        self.__coordinate_system = coord_system

        super(Data, self).__init__(path=path, reader=self.__reader, remap=remap)

    def makeMovie(
        self,
        plot: Callable,
        time: list[float] | None = None,
        num_cpus: int | None = None,
        **movie_kwargs: Any,
    ) -> bool:
        f"""Create animation with provided plot function.
        
        Parameters
        ----------
        plot : callable
            A function that takes a single argument (time in physical units) and produces a plot.
        time : array_like, optional
            An array of time values to use for the animation. If not provided, the entire time range will be used.
        
        Returns
        -------
        bool
            True if the movie was created successfully, False otherwise.
        """
        if time is None:
            if self.fields_defined:
                time = self.fields.t.values
            elif self.particles_defined and self.particles is not None:
                time = list(self.particles.times)
            else:
                raise ValueError("No time values found.")
        assert time is not None, "Time values must be provided."
        name: str = ""
        if self.attrs.get("simulation.name", None) == None:
            name = movie_kwargs.pop("name", "movie")
        else:
            name_b = self.attrs.get("simulation.name")
            if isinstance(name_b, bytes):
                name = name_b.decode("utf-8")
            else:
                name = str(name_b)
        return makeFramesAndMovie(
            name=name,
            data=self,
            plot=plot,
            times=time,
            num_cpus=num_cpus,
            **movie_kwargs,
        )

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """CoordinateSystem: The coordinate system of the data."""
        return self.__coordinate_system

    @property
    def attrs(self) -> dict[str, Any]:
        """dict[str, Any]: The attributes of the data."""
        return self.__attrs

    def to_str(self) -> str:
        """str: String representation of the all the enclosed dataframes."""

        def compactify(lst: list[Any] | KeysView[Any]) -> str:
            c = ""
            cntr = 0
            for l_ in lst:
                if cntr > 5:
                    c += "\n                "
                    cntr = 0
                c += f"{l_}, "
                cntr += 1
            return c[:-2]

        string = ""
        if self.fields_defined:
            string += "Fields:\n"
            string += f"  - coordinates: {self.coordinate_system.value}\n"
            string += f"  - data axes: {compactify(self.fields.indexes.keys())}\n"
            delta_t = (
                self.fields.coords["t"].values[1] - self.fields.coords["t"].values[0]
            ) / (self.fields.coords["s"].values[1] - self.fields.coords["s"].values[0])
            string += f"    - dt: {delta_t:.2e}\n"
            for key in self.fields.coords.keys():
                crd = self.fields.coords[key].values
                fmt = ""
                if key != "s":
                    fmt = ".2f"
                string += f"    - {key}: {crd.min():{fmt}} -> {crd.max():{fmt}} [{len(crd)}]\n"
            string += (
                f"  - quantities: {compactify(sorted(self.fields.data_vars.keys()))}\n"
            )
            string += f"  - total size: {ToHumanReadable(self.fields.nbytes)}\n\n"
        else:
            string += "Fields: empty\n\n"
        if self.particles_defined and self.particles is not None:
            species = sorted(self.particles.species)
            string += "Particle species:\n"
            string += f"  - species: {compactify(species)}\n"
            string += f"  - timesteps: {len(self.particles.times)}\n"
            string += f"  - quantities: {compactify(self.particles.columns)}\n"
            string += f"  - total size: {ToHumanReadable(self.particles.nbytes)}\n\n"
        else:
            string += "Particles: empty\n\n"

        return string

    @override
    def __str__(self) -> str:
        return self.to_str()

    @override
    def __repr__(self) -> str:
        return self.to_str()
