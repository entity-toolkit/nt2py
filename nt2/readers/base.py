from concurrent.futures import as_completed
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import os
import re
import logging

import numpy.typing as npt
from loky import get_reusable_executor
from tqdm import tqdm

from nt2.utils import Format, Layout


def _check_file(enter_file: Callable[[str], Any], filename: str) -> None:
    with enter_file(filename):
        pass


def _get_field_shapes(
    reader: "BaseReader", path: str, step: int
) -> Dict[str, Tuple[int, ...]]:
    names = reader.ReadCategoryNamesAtTimestep(
        path=path,
        category="fields",
        prefix="f",
        step=step,
    )
    return {
        name: reader.ReadArrayShapeAtTimestep(
            path=path,
            category="fields",
            quantity=name,
            step=step,
        )
        for name in names
    }


def _verify_particle_shapes(reader: "BaseReader", path: str, step: int) -> None:
    prtl_species = reader.ReadParticleSpeciesAtTimestep(path=path, step=step)
    quantities = reader.ReadCategoryNamesAtTimestep(
        path=path,
        category="particles",
        prefix="p",
        step=step,
    )
    quantities = set(q.split("_")[0] for q in quantities if q.startswith("p"))
    for species in prtl_species:
        shape = None
        for quantity in quantities:
            current_shape = reader.ReadArrayShapeAtTimestep(
                path=path,
                category="particles",
                quantity=f"{quantity}_{species}",
                step=step,
            )
            if shape is None:
                shape = current_shape
            elif shape != current_shape:
                raise ValueError(
                    f"Different particle shapes found in the {reader.format.value} files for species {species} and quantity {quantity} in step {step}"
                )


class BaseReader:
    """Base virtual class for arbitrary format readers.

    Implements common methods for reading files in different formats and declares virtual methods to be implemented in subclasses.

    """

    skipped_files: List[str]

    def __init__(self) -> None:
        """Initializer for the BaseReader class."""
        self.skipped_files = []

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
        valid_files: List[str],
    ) -> Dict[str, npt.NDArray[Any]]:
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
        valid_files : list[str]
            The valid files to be read.

        Returns
        -------
        dict[str, NDArray[Any]]
            A dictionary with the new name and the variable at each timestep.

        """
        raise NotImplementedError("ReadPerTimestepVariable is not implemented")

    def ReadParticleCountsAtTimestep(
        self,
        path: str,
        step: int,
        species: List[int],
    ) -> Dict[int, int]:
        """Return particle counts by species without reading particle arrays.

        Readers may override this method to collect all counts while opening the
        timestep only once.  The default implementation uses array-shape
        metadata and is kept for third-party readers.
        """
        counts: Dict[int, int] = {}
        for sp in species:
            try:
                counts[sp] = int(
                    self.ReadArrayShapeAtTimestep(
                        path=path,
                        category="particles",
                        quantity=f"pX1_{sp}",
                        step=step,
                    )[0]
                )
            except (IndexError, KeyError, OSError, ValueError):
                counts[sp] = 0
        return counts

    def ReadAttrsAtTimestep(
        self,
        path: str,
        category: str,
        step: int,
    ) -> Dict[str, Any]:
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

    def ReadEdgeCoordsAtTimestep(
        self,
        path: str,
        step: int,
    ) -> Dict[str, npt.NDArray[Any]]:
        """Read the coordinates of cell edges at a given timestep.

        Parameters
        ----------
        path : str
            The path to the files.
        step : int
            The timestep to be read.

        Returns
        -------
        dict[str, NDArray[Any]]
            A dictionary with the coordinates of the cell edges.

        """
        raise NotImplementedError("ReadEdgeCoordsAtTimestep is not implemented")

    def ReadArrayAtTimestep(
        self,
        path: str,
        category: str,
        quantity: str,
        step: int,
    ) -> npt.NDArray[Any]:
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
        NDArray[Any]
            The array at a given timestep.

        """
        raise NotImplementedError("ReadArrayAtTimestep is not implemented")

    def ReadCategoryNamesAtTimestep(
        self,
        path: str,
        category: str,
        prefix: str,
        step: int,
    ) -> Set[str]:
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

    def ReadParticleSpeciesAtTimestep(self, path: str, step: int) -> Set[int]:
        """Read the particle species indices at a given timestep.

        Parameters
        ----------
        path : str
            The path to the files.
        step : int
            The timestep to be read.

        Returns
        -------
        set[int]
            A set of particle species indices at a given timestep.

        """
        return set(
            int(f.split("_")[1])
            for f in self.ReadCategoryNamesAtTimestep(path, "particles", "p", step)
        )

    def ReadArrayShapeAtTimestep(
        self,
        path: str,
        category: str,
        quantity: str,
        step: int,
    ) -> Tuple[int, ...]:
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

    def ReadArrayShapeExplicitlyAtTimestep(
        self,
        path: str,
        category: str,
        quantity: str,
        step: int,
    ) -> Tuple[int, ...]:
        """Read the shape of an array at a given timestep, without relying on metadata.

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

        raise NotImplementedError(
            "ReadArrayShapeExplicitlyAtTimestep is not implemented"
        )

    def ReadFieldCoordsAtTimestep(
        self,
        path: str,
        step: int,
    ) -> Dict[str, npt.NDArray[Any]]:
        """Read the coordinates of the fields at a given timestep.

        Parameters
        ----------
        path : str
            The path to the files.
        step : int
            The timestep to be read.

        Returns
        -------
        dict[str, NDArray[Any]]
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
    def CategoryFiles(path: str, category: str, format: str) -> List[str]:
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
        num_cpus: Optional[int] = None,
    ) -> List[int]:
        """Get valid timesteps (sorted) in a given path and category.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.
        num_cpus : Optional[int]
            The number of CPU cores to use for parallel processing.

        Returns
        -------
        list[int]
            A list of valid timesteps in the given path and category.

        """
        category_files = BaseReader.CategoryFiles(
            path=path,
            category=category,
            format=self.format.value,
        )
        num_cpus = num_cpus if num_cpus is not None else (os.cpu_count() or 1)
        executor = get_reusable_executor(max_workers=num_cpus)
        futures = {
            executor.submit(
                _check_file,
                self.EnterFile,
                os.path.join(path, category, filename),
            ): filename
            for filename in category_files
        }

        steps: List[int] = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"getting valid steps for {category}",
            leave=False,
        ):
            filename = futures[future]
            try:
                future.result()
                steps.append(int(filename.split(".")[1]))
            except OSError:
                if filename not in self.skipped_files:
                    self.skipped_files.append(filename)
                    logging.warning(f"Could not read {filename}, skipping it")
            except Exception as e:
                raise e
        steps.sort()
        return steps

    def GetValidFiles(
        self,
        path: str,
        category: str,
        num_cpus: Optional[int] = None,
    ) -> List[str]:
        """Get valid files (sorted by timestep) in a given path and category.

        Parameters
        ----------
        path : str
            The path to the files.
        category : str
            The category of the files.
        num_cpus : Optional[int]
            The number of CPU cores to use for parallel processing.

        Returns
        -------
        list[str]
            A list of valid files in the given path and category.

        """
        category_files = BaseReader.CategoryFiles(
            path=path,
            category=category,
            format=self.format.value,
        )
        num_cpus = num_cpus if num_cpus is not None else (os.cpu_count() or 1)
        executor = get_reusable_executor(max_workers=num_cpus)
        futures = {
            executor.submit(
                _check_file,
                self.EnterFile,
                os.path.join(path, category, filename),
            ): filename
            for filename in category_files
        }

        files: List[str] = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"getting valid files for {category}",
            leave=False,
        ):
            filename = futures[future]
            try:
                future.result()
                files.append(filename)
            except OSError:
                if filename not in self.skipped_files:
                    self.skipped_files.append(filename)
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
        valid_steps: List[int],
        num_cpus: Optional[int] = None,
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
        valid_steps : list[int]
            The valid timesteps to be checked.
        num_cpus : Optional[int]
            The number of CPU cores to use for parallel processing.

        Raises
        ------
        ValueError
            If different names are found.

        """
        num_cpus = num_cpus if num_cpus is not None else (os.cpu_count() or 1)
        executor = get_reusable_executor(max_workers=num_cpus)
        futures = {
            executor.submit(
                self.ReadCategoryNamesAtTimestep,
                path=path,
                category=category,
                prefix=prefix,
                step=step,
            ): step
            for step in valid_steps
        }
        names_by_step = {}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"verifying same names for {category}",
            leave=False,
        ):
            names_by_step[futures[future]] = future.result()

        names = None
        for step in valid_steps:
            if names is None:
                names = names_by_step[step]
            elif names != names_by_step[step]:
                raise ValueError(
                    f"Different field names found in the {self.format.value} files for step {step}"
                )

    def VerifySameFieldShapes(
        self,
        path: str,
        valid_steps: List[int],
        num_cpus: Optional[int] = None,
    ):
        """Verify that all fields in a given path have the same shape.

        Parameters
        ----------
        path : str
            The path to the files.
        valid_steps : list[int]
            The valid timesteps to be checked.
        num_cpus : Optional[int]
            The number of CPU cores to use for parallel processing.

        Raises
        ------
        ValueError
            If different shapes are found.

        """
        num_cpus = num_cpus if num_cpus is not None else (os.cpu_count() or 1)
        executor = get_reusable_executor(max_workers=num_cpus)
        futures = {
            executor.submit(_get_field_shapes, self, path, step): step
            for step in valid_steps
        }
        shapes_by_step = {}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="verifying same shapes for fields",
            leave=False,
        ):
            shapes_by_step[futures[future]] = future.result()

        shape = None
        for step in valid_steps:
            names = set(shapes_by_step[step])
            if shape is None:
                name = names.pop()
                shape = shapes_by_step[step][name]
            for name in names:
                if shape != shapes_by_step[step][name]:
                    raise ValueError(
                        f"Different field shapes found in the {self.format.value} files for field {name} in step {step}"
                    )

    def VerifySameFieldLayouts(
        self,
        path: str,
        valid_steps: List[int],
        num_cpus: Optional[int] = None,
    ):
        """Verify that all timesteps in a given path have the same layout.

        Parameters
        ----------
        path : str
            The path to the files.
        valid_steps : list[int]
            The valid timesteps to be checked.
        num_cpus : Optional[int]
            The number of CPU cores to use for parallel processing.

        Raises
        ------
        ValueError
            If different layouts are found.

        """
        num_cpus = num_cpus if num_cpus is not None else (os.cpu_count() or 1)
        executor = get_reusable_executor(max_workers=num_cpus)
        futures = {
            executor.submit(
                self.ReadFieldLayoutAtTimestep,
                path=path,
                step=step,
            ): step
            for step in valid_steps
        }
        layouts_by_step = {}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="verifying same layouts for fields",
            leave=False,
        ):
            layouts_by_step[futures[future]] = future.result()

        layout = None
        for step in valid_steps:
            if layout is None:
                layout = layouts_by_step[step]
            elif layout != layouts_by_step[step]:
                raise ValueError(
                    f"Different field layouts found in the {self.format.value} files for step {step}"
                )

    def VerifySameParticleShapes(
        self,
        path: str,
        valid_steps: List[int],
        num_cpus: Optional[int] = None,
    ):
        """Verify that all particle quantities in a given path have the same shape at specific timesteps.

        Parameters
        ----------
        path : str
            The path to the files.
        valid_steps : list[int]
            The valid timesteps to be checked.
        num_cpus : Optional[int]
            The number of CPU cores to use for parallel processing.

        Raises
        ------
        ValueError
            If different shapes are found.

        """
        num_cpus = num_cpus if num_cpus is not None else (os.cpu_count() or 1)
        executor = get_reusable_executor(max_workers=num_cpus)
        futures = [
            executor.submit(_verify_particle_shapes, self, path, step)
            for step in valid_steps
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="verifying same shapes for particles",
            leave=False,
        ):
            future.result()

    def DefinesCategory(self, path: str, category: str, valid_files: List[str]) -> bool:
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
        return os.path.exists(os.path.join(path, category)) and (len(valid_files) > 0)
