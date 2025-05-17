from typing import Any

import xarray as xr
import dask
import dask.array as da

from nt2.utils import CategorySteps, Format
import nt2.readers.hdf5 as hdf5


class BP5NotImplementedError(Exception):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "BP5 format is not implemented"


class Fields:
    """
    Class to read fields from the simulation

    Parameters
    ----------
    path : str
        Path to the simulation directory

    format : Format
        Format of the simulation files. Default is HDF5.

    Methods
    -------
    read(step: int, field: str) -> Any
        Read a field at a given timestep
    """

    def __init__(self, path: str, format: Format = Format.HDF5):
        """
        Args
        ----
        path : str
            Path to the simulation directory

        format : Format
            Format of the simulation files. Default is HDF5.
        """

        self.__path = path
        self.__format = format

        # find all timesteps
        self.steps = CategorySteps(
            path=self.path,
            category="fields",
            format=self.format.value,
        )
        if len(self.steps) == 0:
            raise ValueError("No field steps found")

        # assign reader functions
        if self.format == Format.HDF5:
            read_per_timestep_variable = hdf5.ReadPerTimestepVariable
            read_category_names = hdf5.ReadCategoryNames
            read_field_coords = hdf5.ReadFieldCoords
            read_field_shape = hdf5.ReadFieldShape
            verify_same_category_names = hdf5.VerifySameCategoryNames
            verify_same_field_shapes = hdf5.VerifySameFieldShapes
        elif self.format == Format.BP5:
            raise BP5NotImplementedError()
        else:
            raise ValueError(f"Unknown format: {self.format}")

        # find all times
        self.times = read_per_timestep_variable(
            path=self.path,
            category="fields",
            varname="Time",
            newname="t",
        )

        # find all field names & verify
        self.names = read_category_names(
            path=self.path,
            category="fields",
            prefix="f",
            step=self.steps[0],
        )
        if len(self.names) == 0:
            raise ValueError("No field names found")
        verify_same_category_names(
            path=self.path,
            category="fields",
        )

        # find field shape & verify
        self.shape = read_field_shape(
            path=self.path,
            quantity=self.names[0],
            step=self.steps[0],
        )
        verify_same_field_shapes(
            path=self.path,
        )

        # find field coordinates & verify
        self.coords = read_field_coords(
            path=self.path,
            step=self.steps[0],
        )
        if len(self.coords) == 0:
            raise ValueError("No field coordinates found")

        all_coords = {**self.times, **self.coords}

        self.dataset = xr.Dataset()
        for name in self.names:
            self.dataset[name] = xr.DataArray(
                data=da.stack(
                    [
                        da.from_delayed(
                            self.read(step=step, field=name),
                            shape=self.shape,
                            dtype="float",
                        )
                        for step in self.steps
                    ],
                    axis=0,
                ),
                dims=all_coords.keys(),
                coords=all_coords,
            )

    @property
    def path(self) -> str:
        return self.__path

    @property
    def format(self) -> Format:
        return self.__format

    @dask.delayed
    def read(self, step: int, field: str) -> Any:
        """
        Read a field at a given timestep

        Args
        ----
        step : int
            Timestep to read

        field : str
            Field to read
        """

        if self.format == Format.HDF5:
            read_array_at_timestep = hdf5.ReadArrayAtTimestep
        elif self.format == Format.BP5:
            raise BP5NotImplementedError()
        else:
            raise ValueError(f"Unknown format: {self.format}")

        return read_array_at_timestep(
            path=self.path,
            category="fields",
            quantity=field,
            step=step,
        )
