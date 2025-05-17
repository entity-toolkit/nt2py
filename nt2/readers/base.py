from typing import Any
import os, re, logging, numpy as np

from nt2.utils import Format, Layout


class BaseReader:
    """Base virtual class for arbitrary format readers.

    Implements common methods for reading files in different formats and declares virtual methods to be implemented in subclasses.

    """

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Virtual methods (to be implemented in subclasses)
    # # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def format(self) -> Format:
        """Format: the format of the reader."""
        raise NotImplementedError("format is not implemented")

    @staticmethod
    def EnterFile(
        filename: str,
    ) -> Any:
        """Open a file and return the file object.

        Parameters
        ----------
        filename: str
            The full path to the file to be opened.

        Returns
        -------
        Any
            A file object.

        """
        raise NotImplementedError("EnterFile is not implemented")

    def ReadPerTimestepVariable(
        self,
        path: str,
        category: str,
        varname: str,
        newname: str,
    ) -> dict[str, np.ndarray]:
        """Read a variable at each timestep and return a dictionary with the new name.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.
        varname : str
            The name of the variable to be read.
        newname : str
            The new name of the variable to be returned.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary with the new name and the variable at each timestep.

        """
        raise NotImplementedError("ReadPerTimestepVariable is not implemented")

    def ReadAttrsAtTimestep(
        self,
        path: str,
        category: str,
        step: int,
    ) -> dict[str, Any]:
        """Read the attributes of a given timestep.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.
        step : int
            The timestep to be read.

        Returns
        -------
        dict[str, Any]
            A dictionary with the attributes of the timestep.

        """
        raise NotImplementedError("ReadAttrsAtTimestep is not implemented")

    def ReadArrayAtTimestep(
        self,
        path: str,
        category: str,
        quantity: str,
        step: int,
    ) -> Any:
        """Read an array at a given timestep.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.
        quantity : str
            The name of the array to be read.
        step : int
            The timestep to be read.

        Returns
        -------
        Any
            The array at a given timestep.

        """
        raise NotImplementedError("ReadArrayAtTimestep is not implemented")

    def ReadCategoryNamesAtTimestep(
        self,
        path: str,
        category: str,
        prefix: str,
        step: int,
    ) -> set[str]:
        """Read the names of the variables in a given category and timestep.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.
        prefix : str
            The prefix of the variables to be read.
        step : int
            The timestep to be read.

        Returns
        -------
        set[str]
            The names of the variables in the category.

        """
        raise NotImplementedError("ReadCategoryNamesAtTimestep is not implemented")

    def ReadArrayShapeAtTimestep(
        self,
        path: str,
        category: str,
        quantity: str,
        step: int,
    ) -> tuple[int]:
        """Read the shape of an array at a given timestep.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.
        quantity : str
            The name of the quantity to be read.
        step : int
            The timestep to be read.

        Returns
        -------
        tuple[int]
            The shape of the array at a given timestep.

        """
        raise NotImplementedError("ReadArrayShapeAtTimestep is not implemented")

    def ReadFieldCoordsAtTimestep(
        self,
        path: str,
        step: int,
    ) -> dict[str, Any]:
        """Read the coordinates of the fields at a given timestep.

        Parameters
        ----------
        path : str
            The path to the files.
        step : int
            The timestep to be read.

        Returns
        -------
        dict[str, Any]
            A dictionary with the coordinates of the fields where the keys are the names of the coordinates and the values are.

        """
        raise NotImplementedError("ReadFieldCoordsAtTimestep is not implemented")

    def ReadFieldLayoutAtTimestep(self, path: str, step: int) -> Layout:
        """Read the layout of the fields at a given timestep.

        Parameters
        ----------
        path : str
            The path to the files.
        step : int
            The timestep to be read.

        Returns
        -------
        Layout
            The layout of the fields at a given timestep (R or L).

        """
        raise NotImplementedError("ReadFieldLayoutAtTimestep is not implemented")

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Common methods
    # # # # # # # # # # # # # # # # # # # # # # # #

    @staticmethod
    def CategoryFiles(path: str, category: str, format: str) -> list[str]:
        """Get the list of files in a given category and format.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.
        format : str
            The format of the files.

        Returns
        -------
        list[str]
            A list of files in the given category and format.

        Raises
        ------
        ValueError
            If no files are found.

        """
        files = [
            f
            for f in os.listdir(os.path.join(path, category))
            if re.match(rf"^{category}\.\d{{{8}}}\.{format}", f)
        ]
        files.sort(key=lambda x: int(x.split(".")[1]))
        if len(files) == 0:
            raise ValueError(f"No {category} files found in the specified path")
        return files

    def FullPath(self, path: str, category: str, step: int) -> str:
        """Get the full path to a file.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.
        step : int
            The timestep to be read.

        Returns
        -------
        str
            The full path to the file.

        """
        return os.path.join(
            path, category, f"{category}.{step:08d}.{self.format.value}"
        )

    def GetValidSteps(
        self,
        path: str,
        category: str,
    ) -> list[int]:
        """Get valid timesteps (sorted) in a given path and category.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.

        Returns
        -------
        list[int]
            A list of valid timesteps in the given path and category.

        """
        steps = []
        for filename in BaseReader.CategoryFiles(
            path=path,
            category=category,
            format=self.format.value,
        ):
            try:
                with self.EnterFile(os.path.join(path, category, filename)):
                    step = int(filename.split(".")[1])
                    steps.append(step)
            except OSError:
                logging.warning(f"Could not read {filename}, skipping it")
            except Exception as e:
                raise e
        steps.sort()
        return steps

    def GetValidFiles(
        self,
        path: str,
        category: str,
    ) -> list[str]:
        """Get valid files (sorted by timestep) in a given path and category.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.

        Returns
        -------
        list[str]
            A list of valid files in the given path and category.

        """
        files = []
        for filename in BaseReader.CategoryFiles(
            path=path,
            category=category,
            format=self.format.value,
        ):
            try:
                with self.EnterFile(os.path.join(path, category, filename)):
                    files.append(filename)
            except OSError:
                logging.warning(f"Could not read {filename}, skipping it")
            except Exception as e:
                raise e
        files.sort(key=lambda x: int(x.split(".")[1]))
        return files

    def VerifySameCategoryNames(
        self,
        path: str,
        category: str,
        prefix: str,
    ):
        """Verify that all files in a given category have the same names.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.
        prefix : str
            The prefix of the variables to be read.

        Raises
        ------
        ValueError
            If different names are found.

        """
        names = None
        for step in self.GetValidSteps(
            path=path,
            category=category,
        ):
            if names is None:
                names = self.ReadCategoryNamesAtTimestep(
                    path=path,
                    category=category,
                    prefix=prefix,
                    step=step,
                )
            else:
                if names != self.ReadCategoryNamesAtTimestep(
                    path=path,
                    category=category,
                    prefix=prefix,
                    step=step,
                ):
                    raise ValueError(
                        f"Different field names found in the {self.format.value} files for step {step}"
                    )

    def VerifySameFieldShapes(
        self,
        path: str,
    ):
        """Verify that all fields in a given path have the same shape.

        Parameters
        ----------
        path : str
            The path to the files.

        Raises
        ------
        ValueError
            If different shapes are found.

        """
        shape = None
        for step in self.GetValidSteps(
            path=path,
            category="fields",
        ):
            names = self.ReadCategoryNamesAtTimestep(
                path=path,
                category="fields",
                prefix="f",
                step=step,
            )
            if shape is None:
                name = names.pop()
                shape = self.ReadArrayShapeAtTimestep(
                    path=path,
                    category="fields",
                    quantity=name,
                    step=step,
                )
            for name in names:
                if shape != self.ReadArrayShapeAtTimestep(
                    path=path,
                    category="fields",
                    quantity=name,
                    step=step,
                ):
                    raise ValueError(
                        f"Different field shapes found in the {self.format.value} files for field {name} in step {step}"
                    )

    def VerifySameFieldLayouts(self, path: str):
        """Verify that all timesteps in a given path have the same layout.

        Parameters
        ----------
        path : str
            The path to the files.

        Raises
        ------
        ValueError
            If different layouts are found.

        """
        layout = None
        for step in self.GetValidSteps(
            path=path,
            category="fields",
        ):
            if layout is None:
                layout = self.ReadFieldLayoutAtTimestep(
                    path=path,
                    step=step,
                )
            else:
                if layout != self.ReadFieldLayoutAtTimestep(
                    path=path,
                    step=step,
                ):
                    raise ValueError(
                        f"Different field layouts found in the {self.format.value} files for step {step}"
                    )

    def DefinesCategory(self, path: str, category: str) -> bool:
        """Check whether a given category is defined in the path.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category to be checked.

        Returns
        -------
        bool
            True if the category is defined, False otherwise.

        """
        return os.path.exists(os.path.join(path, category)) and (
            len(self.GetValidFiles(path=path, category=category)) > 0
        )
