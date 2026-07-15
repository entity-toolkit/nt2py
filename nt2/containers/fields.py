from typing import Any, Union, Dict

import dask
import dask.array as da
import xarray as xr
from tqdm import tqdm

from .base import BaseContainer
from ..utils import Layout, CoordinateSystem


def remap_fields_cart(name: str) -> str:
    name = name[1:]
    fieldname = name.split("_")[0]
    fieldname = fieldname.replace("0", "t")
    fieldname = fieldname.replace("1", "x")
    fieldname = fieldname.replace("2", "y")
    fieldname = fieldname.replace("3", "z")
    suffix = "_".join(name.split("_")[1:])
    return f"{fieldname}{'_' + suffix if suffix != '' else ''}"


def remap_coords_cart(name: str) -> str:
    return {
        "X1": "x",
        "X2": "y",
        "X3": "z",
    }.get(name, name)


def remap_fields_sph(name: str) -> str:
    name = name[1:]
    fieldname = name.split("_")[0]
    fieldname = fieldname.replace("0", "t")
    fieldname = fieldname.replace("1", "r")
    fieldname = fieldname.replace("2", "th")
    fieldname = fieldname.replace("3", "ph")
    suffix = "_".join(name.split("_")[1:])
    return f"{fieldname}{'_' + suffix if suffix != '' else ''}"


def remap_coords_sph(name: str) -> str:
    return {
        "X1": "r",
        "X2": "th",
        "X3": "ph",
    }.get(name, name)


class FieldContainer(BaseContainer):
    """Parent class to manage the fields dataframe."""

    __fields_defined: bool = False
    __fields: Union[xr.Dataset, None] = None

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
        super().__init__(**kwargs)
        self.valid_files = self.reader.GetValidFiles(
            path=self.path,
            category="fields",
            num_cpus=self.num_cpus,
        )
        self.valid_steps = self.reader.GetValidSteps(
            path=self.path,
            category="fields",
            num_cpus=self.num_cpus,
        )

        if self.reader.DefinesCategory(self.path, "fields", self.valid_files):
            self.__fields_defined = True
            self.__fields = self._read_fields()

    @property
    def fields_defined(self) -> bool:
        """bool: Whether the fields category is defined."""
        return self.__fields_defined

    @property
    def fields(self) -> Union[xr.Dataset, None]:
        """xr.Dataset: The fields dataframe."""
        return self.__fields

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

    def _read_fields(self) -> xr.Dataset:
        """Helper function to read the fields dataframe."""

        # ensure that the category names, shapes, and layouts are consistent across all steps
        if self.verify:
            self.reader.VerifySameCategoryNames(
                self.path,
                "fields",
                "f",
                self.valid_steps,
                self.num_cpus,
            )
            self.reader.VerifySameFieldShapes(
                self.path,
                self.valid_steps,
                self.num_cpus,
            )
            self.reader.VerifySameFieldLayouts(
                self.path,
                self.valid_steps,
                self.num_cpus,
            )

        # read the field names, layout, shape, coordinates, and attributes from the first step
        first_step = self.valid_steps[0]
        field_names = self.reader.ReadCategoryNamesAtTimestep(
            self.path, "fields", "f", first_step
        )
        first_name = next(iter(field_names))
        layout = self.reader.ReadFieldLayoutAtTimestep(self.path, first_step)
        shape = self.reader.ReadArrayShapeAtTimestep(
            self.path, "fields", first_name, first_step
        )
        coords = self.reader.ReadFieldCoordsAtTimestep(self.path, first_step)
        coords = {k: coords[k] for k in sorted(coords.keys())[::-1]}
        attributes = self.reader.ReadAttrsAtTimestep(
            path=self.path, category="fields", step=first_step
        )
        if self.coordinate_system is None:
            if "Coordinates" not in attributes:
                raise ValueError("Coordinates not found in attributes for fields.")
            if attributes["Coordinates"] in [b"cart", "cart"]:
                self.set_coordinate_system(CoordinateSystem.XYZ)
            elif attributes["Coordinates"] in [b"sph", "sph", b"qsph", "qsph"]:
                self.set_coordinate_system(CoordinateSystem.SPH)
            else:
                raise NotImplementedError(
                    f"Coordinate system {attributes['Coordinates']} not supported."
                )

        if self.remap is None:
            self.set_remap(
                {
                    "coords": (
                        remap_coords_cart
                        if self.coordinate_system == CoordinateSystem.XYZ
                        else remap_coords_sph
                    ),
                    "fields": (
                        remap_fields_cart
                        if self.coordinate_system == CoordinateSystem.XYZ
                        else remap_fields_sph
                    ),
                }
            )

        # rename coordinates if remap is provided
        if self.remap is not None and "coords" in self.remap:
            new_coords = {}
            for coord in coords.keys():
                new_coords[self.remap["coords"](coord)] = coords[coord]
            coords = new_coords

        times = self.reader.ReadPerTimestepVariable(
            self.path,
            "fields",
            "Time",
            "t",
            self.valid_files,
        )
        steps = self.reader.ReadPerTimestepVariable(
            self.path,
            "fields",
            "Step",
            "s",
            self.valid_files,
        )

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
                            for step in tqdm(
                                self.valid_steps,
                                desc="steps",
                                position=1,
                                leave=False,
                            )
                        ],
                        axis=0,
                    ),
                    name=remapped_name,
                    dims=all_dims,
                    coords=all_coords,
                )
                for name in tqdm(field_names, desc="fields", position=0, leave=False)
            },
            attrs=attributes,
        )

    @property
    def attrs(self) -> Dict[str, Any]:
        """dict: The attributes of the fields dataframe."""
        if self.fields_defined:
            return self.fields.attrs
        else:
            return {}
