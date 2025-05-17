from typing import Any

import dask
import dask.array as da
import xarray as xr

from nt2.containers.container import BaseContainer
from nt2.readers.base import BaseReader
from nt2.utils import Layout, ToHumanReadable


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
            self.__fields = self.__read_fields()
        else:
            self.__fields = None
        pass

    @property
    def fields(self) -> xr.Dataset | None:
        """xr.Dataset | None: The fields dataframe."""
        return self.__fields

    def to_str(self) -> str:
        """
        Returns
        -------
        str
            String representation of the fields dataframe.

        """

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
        if self.fields is not None:
            field_keys = list(self.fields.data_vars.keys())
            string += "Fields:\n"
            string += f"  - data axes: {compactify(self.fields.indexes.keys())}\n"
            string += f"  - timesteps: {self.fields[field_keys[0]].shape[0]}\n"
            string += f"  - shape: {self.fields[field_keys[0]].shape[1:]}\n"
            string += f"  - quantities: {compactify(self.fields.data_vars.keys())}\n"
            string += f"  - total size: {ToHumanReadable(self.fields.nbytes)}\n"
        else:
            string += "Fields: empty\n"

        return string

    def __read_fields(self):
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

        all_dims = {**times, **coords}.keys()
        all_coords = {**times, **coords, "s": ("t", steps["s"])}

        self.reader.ReadFieldLayoutAtTimestep(self.path, first_step)

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
                    da.stack(  # type: ignore
                        [
                            da.from_delayed(  # type: ignore
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
