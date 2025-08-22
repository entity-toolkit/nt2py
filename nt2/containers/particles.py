from typing import Any

import dask
import dask.array as da
import xarray as xr
import numpy as np

from nt2.containers.container import BaseContainer
from nt2.readers.base import BaseReader


class Particles(BaseContainer):
    """Parent class to manage the particles dataframe."""

    @staticmethod
    @dask.delayed
    def __read_species_quantity(
        path: str, reader: BaseReader, species_quantity: str, step: int, pad: int
    ) -> Any:
        """Reads a species from the data.

        This is a dask-delayed function used further to build the dataset.

        Parameters
        ----------
        path : str
            Main path to the data.
        reader : BaseReader
            Reader to use to read the data.
        species_quantity : str
            Quantity specific to a species to read.
        step : int
            Step to read.
        pad : int
            Length to pad the array to.

        Returns
        -------
        Any
            Species data.

        """
        arr = reader.ReadArrayAtTimestep(path, "particles", species_quantity, step)
        shape = reader.ReadArrayShapeAtTimestep(
            path, "particles", species_quantity, step
        )[0]
        return da.pad(arr, ((0, pad - shape),), mode="constant", constant_values=np.nan)

    def __init__(self, **kwargs: Any) -> None:
        """Initializer for the Particles class.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the parent BaseContainer class.

        """
        super(Particles, self).__init__(**kwargs)
        if (
            self.reader.DefinesCategory(self.path, "particles")
            and self.particles_present
        ):
            self.__particles_defined = True
            self.__particles = self.__read_particles()
        else:
            self.__particles_defined = False
            self.__particles = {}

    @property
    def particles_present(self) -> bool:
        """bool: Whether the particles are present in any of the timesteps."""
        return len(self.nonempty_steps) > 0

    @property
    def nonempty_steps(self) -> list[int]:
        """list[int]: List of timesteps that contain particles data."""
        valid_steps = self.reader.GetValidSteps(self.path, "particles")
        return [
            step
            for step in valid_steps
            if len(
                set(
                    q.split("_")[0]
                    for q in self.reader.ReadCategoryNamesAtTimestep(
                        self.path, "particles", "p", step
                    )
                    if q.startswith("p")
                )
            )
            > 0
        ]

    @property
    def particles_defined(self) -> bool:
        """bool: Whether the particles category is defined."""
        return self.__particles_defined

    @property
    def particles(self) -> dict[int, xr.Dataset]:
        """Returns the particles data.

        Returns
        -------
        dict[int, xr.Dataset]
            Dictionary of datasets for each step.

        """
        return self.__particles

    def __read_particles(self) -> dict[int, xr.Dataset]:
        """Helper function to read all particles data."""
        self.reader.VerifySameCategoryNames(self.path, "particles", "p")
        self.reader.VerifySameParticleShapes(self.path)

        valid_steps = self.nonempty_steps
        prtl_species = self.reader.ReadParticleSpeciesAtTimestep(
            self.path, valid_steps[0]
        )
        prtl_quantities = set(
            q.split("_")[0]
            for q in self.reader.ReadCategoryNamesAtTimestep(
                self.path, "particles", "p", valid_steps[0]
            )
            if q.startswith("p")
        )
        prtl_quantities = sorted(prtl_quantities)

        first_quantity = next(iter(prtl_quantities))
        maxlens = {
            sp: np.max(
                [
                    self.reader.ReadArrayShapeAtTimestep(
                        self.path, "particles", f"{first_quantity}_{sp}", st
                    )
                    for st in valid_steps
                ]
            )
            for sp in prtl_species
        }

        times = self.reader.ReadPerTimestepVariable(self.path, "particles", "Time", "t")
        steps = self.reader.ReadPerTimestepVariable(self.path, "particles", "Step", "s")

        idxs: dict[int, dict[str, np.ndarray]] = {
            sp: {"idx": np.arange(maxlens[sp])} for sp in prtl_species
        }

        all_dims = {sp: {**times, **(idxs[sp])}.keys() for sp in prtl_species}
        all_coords = {
            sp: {**times, **(idxs[sp]), "s": ("t", steps["s"])} for sp in prtl_species
        }

        def remap_quantity(name: str) -> str:
            """
            Remaps the particle quantity name if remap is provided
            """
            if self.remap is not None and "particles" in self.remap:
                return self.remap["particles"](name)
            return name

        def get_quantity(species: int, quantity: str, step: int, maxlen: int) -> Any:
            return Particles.__read_species_quantity(
                self.path, self.reader, f"{quantity}_{species}", step, maxlen
            )

        return {
            sp: xr.Dataset(
                {
                    remap_quantity(quantity): xr.DataArray(
                        da.stack(
                            [
                                da.from_delayed(
                                    get_quantity(sp, quantity, step, maxlens[sp]),
                                    shape=(maxlens[sp],),
                                    dtype="float",
                                )
                                for step in valid_steps
                            ],
                            axis=0,
                        ),
                        name=remap_quantity(quantity),
                        dims=all_dims[sp],
                        coords=all_coords[sp],
                    )
                    for quantity in prtl_quantities
                }
            )
            for sp in prtl_species
        }
