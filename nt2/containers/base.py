from typing import Callable, Optional, Dict, Tuple, List, Union

from ..readers.base import BaseReader
from ..utils import CoordinateSystem


class BaseContainer:
    """Parent container class for holding any category data."""

    __path: str
    __reader: BaseReader
    __verify: bool
    __remap: Optional[Dict[str, Callable[[str], str]]]
    __coordinate_system: Optional[CoordinateSystem]
    __num_cpus: Optional[int]

    valid_steps: List[int] = []
    valid_files: List[str] = []

    def __init__(
        self,
        path: str,
        reader: BaseReader,
        verify: bool = False,
        remap: Union[Dict[str, Callable[[str], str]], None] = None,
        coord_system: Union[CoordinateSystem, None] = None,
        num_cpus: Union[int, None] = None,
    ):
        """Initializer for the BaseContainer class.

        Parameters
        ----------
        path : str
            The path to the data.
        reader : BaseReader
            The reader to be used for reading the data.
        verify : Optional[bool]
            Whether to verify the data. If None, it will use the reader's default.
        remap : Optional[dict[str, Callable[[str], str]]]
            Remap dictionary to use to remap the data names (coords, fields, etc.).
        coord_system : Optional[CoordinateSystem]
            The coordinate system of the data.
        num_cpus : Optional[int]
            The number of CPUs to use for parallel processing. If None, it will use all available CPUs.

        """
        self.__path = path
        self.__reader = reader
        self.__verify = verify
        self.__remap = remap
        self.__coordinate_system = coord_system
        self.__num_cpus = num_cpus

    @property
    def path(self) -> str:
        """str: The main path of the data."""
        return self.__path

    @property
    def reader(self) -> BaseReader:
        """BaseReader: The reader used to read the data."""
        return self.__reader
    
    @property
    def verify(self) -> bool:
        """bool: Whether to verify the data."""
        return self.__verify

    @property
    def remap(self) -> Optional[Dict[str, Callable[[str], str]]]:
        """{ str: (str) -> str } : The coordinate/field remap dictionary."""
        return self.__remap

    @property
    def coordinate_system(self) -> Optional[CoordinateSystem]:
        """CoordinateSystem: The coordinate system of the data."""
        return self.__coordinate_system

    @property
    def num_cpus(self) -> Optional[int]:
        """int: The number of CPUs to use for parallel processing."""
        return self.__num_cpus

    def set_remap(self, remap: Dict[str, Callable[[str], str]]) -> None:
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

    def __dask_tokenize__(self) -> Tuple[str, str, str]:
        """Provide a deterministic Dask token for container instances."""
        return (
            self.__class__.__name__,
            self.__path,
            self.__reader.format.value,
        )
