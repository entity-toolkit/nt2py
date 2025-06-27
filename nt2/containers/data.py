from typing import Callable

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


class Data(Fields):
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
        if reader is None:
            if DetermineDataFormat(path) == Format.HDF5:
                self.__reader = HDF5Reader()
            elif DetermineDataFormat(path) == Format.BP5:
                self.__reader = BP5Reader()
            else:
                raise NotImplementedError(
                    "Only HDF5 & BP5 formats are supported at the moment."
                )
        else:
            if DetermineDataFormat(path) != reader.format:
                raise ValueError(
                    f"Reader format {reader.format} does not match data format {DetermineDataFormat(path)}."
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
                    if (
                        attrs["Coordinates"] == b"cart"
                        or attrs["Coordinates"] == "cart"
                    ):

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

                        coord_system = CoordinateSystem.XYZ

                    elif (
                        attrs["Coordinates"] == b"sph" or attrs["Coordinates"] == "sph"
                    ):

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

                        coord_system = CoordinateSystem.SPH

                    else:
                        raise NotImplementedError(
                            f"Coordinate system {attrs['Coordinates']} not supported."
                        )
                    if remap is None:
                        remap = {
                            "coords": remap_coords,
                            "fields": remap_fields,
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
