from typing import Any

import dask
import dask.array as da
import xarray as xr

from nt2.containers.container import BaseContainer
from nt2.utils import Layout


class Fields(BaseContainer):
    """Parent class to manage the fields dataframe."""

    def _read_field(self, layout: Layout, field: str, step: int) -> Any:
        """Reads a field from the data.

        This is a dask-delayed function used further to build the dataset.

        Parameters
        ----------
        layout : Layout
            Layout of the field.
        field : str
            Field to read.
        step : int
            Step to read.

        Returns
        -------
        Any
            Field data.

        """
        if layout == Layout.L:
            return self.reader.ReadArrayAtTimestep(self.path, "fields", field, step)
        else:
            return self.reader.ReadArrayAtTimestep(self.path, "fields", field, step).T

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initializer for the Fields class.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the parent BaseContainer class.

        """
        super(Fields, self).__init__(**kwargs)
        if self.reader.DefinesCategory(self.path, "fields"):
            self.__fields_defined = True
            self.__fields = self._read_fields()
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

    def _read_fields(self) -> xr.Dataset:
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
        new_edge_coords = {}
        for coord in edge_coords.keys():
            assoc_x = (
                coord[:-1]
                if (self.remap is None or "coords" not in self.remap)
                else self.remap["coords"](coord[:-1])
            )
            new_edge_coords[assoc_x + "_min"] = (assoc_x, edge_coords[coord][:-1])
            new_edge_coords[assoc_x + "_max"] = (assoc_x, edge_coords[coord][1:])
        edge_coords = new_edge_coords

        all_dims = {**times, **coords}.keys()
        all_coords = {**times, **coords, "s": ("t", steps["s"]), **edge_coords}

        return xr.Dataset(
            {
                (
                    remapped_name := (
                        self.remap["fields"](name)
                        if (self.remap is not None and "fields" in self.remap)
                        else name
                    )
                ): xr.DataArray(
                    da.stack(
                        [
                            da.from_delayed(
                                dask.delayed(self._read_field)(layout, name, step),
                                shape=shape[:: -1 if layout == Layout.R else 1],
                                dtype="float",
                            )
                            for step in valid_steps
                        ],
                        axis=0,
                    ),
                    name=remapped_name,
                    dims=all_dims,
                    coords=all_coords,
                )
                for name in field_names
            },
            attrs=self.reader.ReadAttrsAtTimestep(
                path=self.path, category="fields", step=first_step
            ),
        )
