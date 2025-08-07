from typing import Callable
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

from nt2.plotters.polar import (
    _datasetPolarPlotAccessor,
    _polarPlotAccessor,
)

from nt2.plotters.inspect import _datasetInspectPlotAccessor
from nt2.plotters.movie import _moviePlotAccessor


@xr.register_dataset_accessor("polar")
@InheritClassDocstring
class DatasetPolarPlotAccessor(_datasetPolarPlotAccessor):
    pass


@xr.register_dataarray_accessor("polar")
@InheritClassDocstring
class PolarPlotAccessor(_polarPlotAccessor):
    pass


@xr.register_dataset_accessor("inspect")
@InheritClassDocstring
class DatasetInspectPlotAccessor(_datasetInspectPlotAccessor):
    pass


@xr.register_dataarray_accessor("movie")
@InheritClassDocstring
class MoviePlotAccessor(_moviePlotAccessor):
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
        for category in ["fields", "particles", "spectra"]:
            if self.__reader.DefinesCategory(path, category):
                valid_steps = self.__reader.GetValidSteps(path, category)
                if len(valid_steps) == 0:
                    raise ValueError(f"No valid steps found for category {category}.")
                first_step = valid_steps[0]
                attrs = self.__reader.ReadAttrsAtTimestep(path, category, first_step)
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

                    elif attrs["Coordinates"] in [b"sph", "sph"]:

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

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """CoordinateSystem: The coordinate system of the data."""
        return self.__coordinate_system

    def to_str(self) -> str:
        """
        Returns
        -------
        str
            String representation of the all the enclosed dataframes.

        """

        def compactify(lst):
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
            field_keys = list(self.fields.data_vars.keys())
            string += "Fields:\n"
            string += f"  - data axes: {compactify(self.fields.indexes.keys())}\n"
            string += f"  - timesteps: {self.fields[field_keys[0]].shape[0]}\n"
            string += f"  - shape: {self.fields[field_keys[0]].shape[1:]}\n"
            string += f"  - quantities: {compactify(self.fields.data_vars.keys())}\n"
            string += f"  - total size: {ToHumanReadable(self.fields.nbytes)}\n\n"
        else:
            string += "Fields: empty\n\n"
        if self.particles_defined:
            species = sorted(list(self.particles.keys()))
            string += "Particle species:\n"
            string += f"  - species: {compactify(species)}\n"
            string += f"  - timesteps: {len(self.particles[species[0]].t)}\n"
            string += f"  - quantities: {compactify(self.particles[species[0]].data_vars.keys())}\n"
            string += f"  - max # per species: {[self.particles[sp].idx.shape[0] for sp in species]}\n"
            string += f"  - total size: {ToHumanReadable(sum([self.particles[sp].nbytes for sp in species]))}\n\n"
        else:
            string += "Particles: empty\n\n"

        return string

    def __str__(self) -> str:
        return self.to_str()

    def __repr__(self) -> str:
        return self.to_str()
