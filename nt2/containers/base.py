from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from ..readers.base import BaseReader
from ..utils import CoordinateSystem


class BaseContainer:
    """Parent container class for holding any category data."""

    __path: str
    __category: str
    __reader: BaseReader
    __verify: bool
    __timerange: tuple[float | None, float | None] | None
    __steprange: tuple[int | None, int | None] | None
    __remap: dict[str, Callable[[str], str]] | None
    __coordinate_system: CoordinateSystem | None
    __num_cpus: int | None

    __valid_steps: list[int]
    __valid_files: list[str]

    __times: npt.NDArray
    __steps: npt.NDArray

    def __init__(
        self,
        path: str,
        category: str,
        reader: BaseReader,
        verify: bool = False,
        timerange: tuple[float | None, float | None] | None = None,
        steprange: tuple[int | None, int | None] | None = None,
        remap: dict[str, Callable[[str], str]] | None = None,
        coord_system: CoordinateSystem | None = None,
        num_cpus: int | None = None,
    ):
        """Initializer for the BaseContainer class.

        Parameters
        ----------
        path : str
            The path to the data.
        category : str
            The category of the data.
        reader : BaseReader
            The reader to be used for reading the data.
        verify : Optional[bool]
            Whether to verify the data. If None, it will use the reader's default.
        timerange : Optional[tuple[Union[float, None], Union[float, None]]]
            Time range to load. If None, all times will be loaded.
        steprange : Optional[tuple[Union[int, None], Union[int, None]]]
            Step range to load. If None, all steps will be loaded.
        remap : Optional[dict[str, Callable[[str], str]]]
            Remap dictionary to use to remap the data names (coords, fields, etc.).
        coord_system : Optional[CoordinateSystem]
            The coordinate system of the data.
        num_cpus : Optional[int]
            The number of CPUs to use for parallel processing. If None, it will use all available CPUs.

        """
        self.__path = path
        self.__category = category
        self.__reader = reader
        self.__timerange = timerange
        self.__steprange = steprange
        self.__verify = verify
        self.__remap = remap
        self.__coordinate_system = coord_system
        self.__num_cpus = num_cpus

        self.__valid_files, self.__valid_steps = self.__reader.GetValidFilesAndSteps(
            self.path,
            self.category,
            self.steprange,
            self.num_cpus,
        )
        self.__times, self.__steps = self.read_times_and_steps()
        if self.steprange is None:
            self.narrow_timerange()

    @property
    def path(self) -> str:
        """str: The main path of the data."""
        return self.__path

    @property
    def category(self) -> str:
        """str: The category of the data."""
        return self.__category

    @property
    def reader(self) -> BaseReader:
        """BaseReader: The reader used to read the data."""
        return self.__reader

    @property
    def verify(self) -> bool:
        """bool: Whether to verify the data."""
        return self.__verify

    @property
    def remap(self) -> dict[str, Callable[[str], str]] | None:
        """{ str: (str) -> str } : The coordinate/field remap dictionary."""
        return self.__remap

    @property
    def coordinate_system(self) -> CoordinateSystem | None:
        """CoordinateSystem: The coordinate system of the data."""
        return self.__coordinate_system

    @property
    def num_cpus(self) -> int | None:
        """int: The number of CPUs to use for parallel processing."""
        return self.__num_cpus

    @property
    def timerange(self) -> tuple[float | None, float | None] | None:
        """tuple[float | None, float | None]: The time range of the data."""
        return self.__timerange

    @property
    def steprange(self) -> tuple[int | None, int | None] | None:
        """tuple[int | None, int | None]: The step range of the data."""
        return self.__steprange

    @property
    def valid_files(self) -> list[str]:
        """list[str]: The valid files of the data."""
        return self.__valid_files

    @property
    def valid_steps(self) -> list[int]:
        """list[int]: The valid steps of the data."""
        return self.__valid_steps

    @property
    def times(self) -> npt.NDArray:
        """npt.NDArray: The times of the data."""
        return self.__times

    @property
    def steps(self) -> npt.NDArray:
        """npt.NDArray: The steps of the data."""
        return self.__steps

    def read_times_and_steps(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Reads the times and steps for the given category.

        Parameters
        ----------
        category : str
            The category to read the times and steps for.

        Returns
        -------
        tuple[npt.NDArray, npt.NDArray]
            A tuple containing the times and steps for the given category.

        """
        vars = self.reader.ReadPerTimestepVariables(
            path=self.path,
            category=self.category,
            varnames=["Time", "Step"],
            newnames=["t", "s"],
            valid_files=self.valid_files,
        )
        return (vars["t"], vars["s"])

    def narrow_timerange(self):
        if self.timerange is not None:
            start_time, end_time = self.timerange
            start_idx, end_idx = 0, -1
            if start_time is None:
                start_idx = 0
            else:
                start_idx = int(np.searchsorted(self.__times, start_time))
            if end_time is None:
                end_idx = len(self.__times) - 1
            else:
                end_idx = int(np.searchsorted(self.__times, end_time))
            self.__times = self.__times[start_idx : end_idx + 1]
            self.__steps = self.__steps[start_idx : end_idx + 1]
            self.__valid_files = self.__valid_files[start_idx : end_idx + 1]
            self.__valid_steps = self.__valid_steps[start_idx : end_idx + 1]

    def set_remap(self, remap: dict[str, Callable[[str], str]]) -> None:
        """Set the remap dictionary for the container.

        Parameters
        ----------
        remap : dict[str, Callable[[str], str]]
            The remap dictionary to set.

        """
        self.__remap = remap

    def set_coordinate_system(self, coord_system: CoordinateSystem) -> None:
        """Set the coordinate system of the data.

        Parameters
        ----------
        coord_system : CoordinateSystem
            The coordinate system to set.

        """
        self.__coordinate_system = coord_system

    def __dask_tokenize__(self) -> tuple[str, str, str]:
        """Provide a deterministic Dask token for container instances."""
        return (
            self.__class__.__name__,
            self.__path,
            self.__reader.format.value,
        )
