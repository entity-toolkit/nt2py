from typing import Callable

from nt2.readers.base import BaseReader


class BaseContainer:
    """Parent container class for holding any category data."""

    def __init__(
        self,
        path: str,
        reader: BaseReader,
        remap: dict[str, Callable[[str], str]] | None = None,
    ):
        """Initializer for the BaseContainer class.

        Parameters
        ----------
        path : str
            The path to the data.
        reader : BaseReader
            The reader to be used for reading the data.
        remap : dict[str, Callable[[str], str]] | None
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
    def remap(self) -> dict[str, Callable[[str], str]] | None:
        """dict[str, Callable[[str], str]]: The coordinate/field remap dictionary."""
        return self.__remap
