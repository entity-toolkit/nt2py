from typing import Any

import dask
import dask.array as da
import xarray as xr

from nt2.containers.container import BaseContainer
from nt2.readers.base import BaseReader
from nt2.utils import Layout


class Fields(BaseContainer):
    """Parent class to manage the fields dataframe."""

    @staticmethod
    @dask.delayed  # type: ignore
    def __read_field(path: str, reader: BaseReader, field: str, step: int) -> Any:
        """Reads a field from the data.

        This is a dask-delayed function used further to build the dataset.

        Parameters
        ----------
        path : str
            Main path to the data.
        reader : BaseReader
            Reader to use to read the data.
        field : str
            Field to read.
        step : int
            Step to read.

        Returns
        -------
        Any
            Field data.

        """
        return reader.ReadArrayAtTimestep(path, "fields", field, step)

    def __init__(
        self,
        **kwargs,
    ):
        """Initializer for the Fields class.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the parent BaseContainer class.

        """
        super(Fields, self).__init__(**kwargs)
        if self.reader.DefinesCategory(self.path, "fields"):
            self.__fields_defined = True
            self.__fields = self.__read_fields()
        else:
            self.__fields_defined = False
            self.__fields = xr.Dataset()

    @property
    def fields_defined(self) -> bool:
        """bool: Whether the fields category is defined."""
        return self.__fields_defined

    @property
    def fields(self) -> xr.Dataset:
        """xr.Dataset: The fields dataframe."""
        return self.__fields

    def __read_fields(self) -> xr.Dataset:
        """Helper function to read the fields dataframe."""
        self.reader.VerifySameCategoryNames(self.path, "fields", "f")
        self.reader.VerifySameFieldShapes(self.path)
        self.reader.VerifySameFieldLayouts(self.path)

        valid_steps = self.reader.GetValidSteps(self.path, "fields")
        field_names = self.reader.ReadCategoryNamesAtTimestep(
            self.path, "fields", "f", valid_steps[0]
        )

        first_step = valid_steps[0]
        first_name = next(iter(field_names))
        layout = self.reader.ReadFieldLayoutAtTimestep(self.path, first_step)
        shape = self.reader.ReadArrayShapeAtTimestep(
            self.path, "fields", first_name, first_step
        )
        coords = self.reader.ReadFieldCoordsAtTimestep(self.path, first_step)
        coords = {k: coords[k] for k in sorted(coords.keys())[::-1]}
        # rename coordinates if remap is provided
        if self.remap is not None and "coords" in self.remap:
            new_coords = {}
            for coord in coords.keys():
                new_coords[self.remap["coords"](coord)] = coords[coord]
            coords = new_coords

        times = self.reader.ReadPerTimestepVariable(self.path, "fields", "Time", "t")
        steps = self.reader.ReadPerTimestepVariable(self.path, "fields", "Step", "s")

        edge_coords = self.reader.ReadEdgeCoordsAtTimestep(self.path, first_step)
        if self.remap is not None and "coords" in self.remap:
            new_edge_coords = {}
            for coord in edge_coords.keys():
                assoc_x = self.remap["coords"](coord[:-1])
                new_edge_coords[assoc_x + "_min"] = (assoc_x, edge_coords[coord][:-1])
                new_edge_coords[assoc_x + "_max"] = (assoc_x, edge_coords[coord][1:])
            edge_coords = new_edge_coords

        all_dims = {**times, **coords}.keys()
        all_coords = {**times, **coords, "s": ("t", steps["s"]), **edge_coords}

        def remap_name(name: str) -> str:
            """
            Remaps the field name if remap is provided
            """
            if self.remap is not None and "fields" in self.remap:
                return self.remap["fields"](name)
            return name

        def get_field(name: str, step: int) -> Any:
            """
            Reads a field from the data
            """
            if layout == Layout.L:
                return Fields.__read_field(self.path, self.reader, name, step)
            else:
                return Fields.__read_field(self.path, self.reader, name, step).T

        return xr.Dataset(
            {
                remap_name(name): xr.DataArray(
                    da.stack(
                        [
                            da.from_delayed(
                                get_field(name, step),
                                shape=shape[:: -1 if layout == Layout.R else 1],
                                dtype="float",
                            )
                            for step in valid_steps
                        ],
                        axis=0,
                    ),
                    name=remap_name(name),
                    dims=all_dims,
                    coords=all_coords,
                )
                for name in field_names
            },
            attrs=self.reader.ReadAttrsAtTimestep(
                path=self.path, category="fields", step=first_step
            ),
        )
