from typing import Callable, Optional, Dict, Tuple

from nt2.readers.base import BaseReader


class BaseContainer:
    """Parent container class for holding any category data."""

    __path: str
    __reader: BaseReader
    __remap: Optional[Dict[str, Callable[[str], str]]]

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

    def __dask_tokenize__(self) -> Tuple[str, str, str]:
        """Provide a deterministic Dask token for container instances."""
        return (
            self.__class__.__name__,
            self.__path,
            self.__reader.format.value,
        )
