from typing import Callable, Optional, Dict, Tuple, List

from nt2.readers.base import BaseReader


class BaseContainer:
    """Parent container class for holding any category data."""

    __path: str
    __reader: BaseReader
    __remap: Optional[Dict[str, Callable[[str], str]]]

    __valid_steps: Dict[str, List[int]]
    __valid_files: Dict[str, List[str]]

    def __init__(
        self,
        path: str,
        reader: BaseReader,
        remap: Optional[Dict[str, Callable[[str], str]]] = None,
    ):
        """Initializer for the BaseContainer class.

        Parameters
        ----------
        path : str
            The path to the data.
        reader : BaseReader
            The reader to be used for reading the data.
        remap : Optional[dict[str, Callable[[str], str]]]
            Remap dictionary to use to remap the data names (coords, fields, etc.).

        """
        super(BaseContainer, self).__init__()
        self.__path = path
        self.__reader = reader
        self.__remap = remap

    @property
    def path(self) -> str:
        """str: The main path of the data."""
        return self.__path

    @property
    def reader(self) -> BaseReader:
        """BaseReader: The reader used to read the data."""
        return self.__reader

    @property
    def remap(self) -> Optional[Dict[str, Callable[[str], str]]]:
        """{ str: (str) -> str } : The coordinate/field remap dictionary."""
        return self.__remap

    @property
    def valid_steps(self) -> Dict[str, List[int]]:
        """dict[str, list[int]]: The valid steps for each category."""
        return self.__valid_steps

    @property
    def valid_files(self) -> Dict[str, List[str]]:
        """dict[str, list[str]]: The valid files for each category."""
        return self.__valid_files

    @valid_steps.setter
    def valid_steps(self, value: Dict[str, List[int]]) -> None:
        """Set the valid steps for each category.

        Parameters
        ----------
        value : dict[str, list[int]]
            The valid steps for each category.

        """
        self.__valid_steps = value

    @valid_files.setter
    def valid_files(self, value: Dict[str, List[str]]) -> None:
        """Set the valid files for each category.

        Parameters
        ----------
        value : dict[str, list[str]]
            The valid files for each category.

        """
        self.__valid_files = value

    def __dask_tokenize__(self) -> Tuple[str, str, str]:
        """Provide a deterministic Dask token for container instances."""
        return (
            self.__class__.__name__,
            self.__path,
            self.__reader.format.value,
        )
