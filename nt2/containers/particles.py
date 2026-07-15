from typing import Any, List, Union, Dict, Set
import numpy.typing as npt

import numpy as np

from .base import BaseContainer
from .particle_dataset import ParticleDataset
from ..utils import CoordinateSystem


def remap_prtl_quantities_cart(name: str) -> str:
    shortname = name[1:]
    return {
        "X1": "x",
        "X2": "y",
        "X3": "z",
        "U1": "ux",
        "U2": "uy",
        "U3": "uz",
        "W": "w",
    }.get(shortname, shortname)


def remap_prtl_quantities_sph(name: str) -> str:
    shortname = name[1:]
    return {
        "X1": "r",
        "X2": "th",
        "X3": "ph",
        "U1": "ur",
        "U2": "uth",
        "U3": "uph",
        "W": "w",
    }.get(shortname, shortname)


class ParticleContainer(BaseContainer):
    """Parent class to manage the particles dataframe."""

    __particles_defined: bool = False
    __particles: Union[ParticleDataset, None] = None

    nonempty_steps: List[int]
    attributes: Dict[str, Any]
    quantities: List[str]
    sp_with_idx: List[int]
    sp_without_idx: List[int]
    quantity_names_by_step: Dict[int, Set[str]]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        particles = state.get("_ParticleContainer__particles")
        if particles is not None:
            state["_ParticleContainer__particle_dataset_state"] = {
                "species": particles.species,
                "steps": particles.steps,
                "times": particles.times,
                "colnames": particles.colnames,
                "fprec": particles.fprec,
                "selection": particles.selection,
                "partition_lengths": particles._partition_lengths,
            }
        state.pop("_ParticleContainer__particles", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        particle_dataset_state = state.pop(
            "_ParticleContainer__particle_dataset_state", None
        )
        self.__dict__.update(state)
        if self.__particles_defined:
            if particle_dataset_state is None:
                (
                    self.quantities,
                    self.sp_with_idx,
                    self.sp_without_idx,
                    self.attributes,
                    self.__particles,
                ) = self._read_particles()
            else:
                self.__particles = ParticleDataset(
                    **particle_dataset_state,
                    read_column=self._read_column,
                )

    def __init__(self, **kwargs: Any) -> None:
        """Initializer for the ParticleContainer class.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the parent BaseContainer class.

        """
        super().__init__(category="particles", **kwargs)

        # @TODO: parallelize
        self.quantity_names_by_step = {
            step: self.reader.ReadCategoryNamesAtTimestep(
                self.path, "particles", "p", step
            )
            for step in self.valid_steps
        }
        self.nonempty_steps = [
            step
            for step, names in self.quantity_names_by_step.items()
            if any(q.startswith("p") for q in names)
        ]

        if (
            self.reader.DefinesCategory(self.path, "particles", self.valid_files)
            and len(self.nonempty_steps) > 0
        ):
            self.__particles_defined = True
            (
                self.quantities,
                self.sp_with_idx,
                self.sp_without_idx,
                self.attributes,
                self.__particles,
            ) = self._read_particles()

    def _read_particles(self):
        # read unique quantities and species
        quantities_ = [
            self.quantity_names_by_step[step] for step in self.nonempty_steps
        ]
        quantities = sorted(np.unique([q for qtys in quantities_ for q in qtys]))

        unique_quantities = sorted(
            list(
                set(
                    f"{q}".split("_")[0]
                    for q in quantities
                    if not q.startswith("pIDX") and not q.startswith("pRNK")
                )
            )
        )
        all_species = sorted(list(set([int(f"{q}".split("_")[1]) for q in quantities])))

        sp_with_idx = sorted(
            [int(f"{q}".split("_")[1]) for q in quantities if f"{q}".startswith("pIDX")]
        )
        sp_without_idx = sorted([sp for sp in all_species if sp not in sp_with_idx])

        partition_lengths = tuple(
            sum(
                self.reader.ReadParticleCountsAtTimestep(
                    self.path, step, all_species
                ).values()
            )
            for step in self.valid_steps
        )

        # determine coordinate system and remap functions
        first_step = self.valid_steps[0]
        attributes = self.reader.ReadAttrsAtTimestep(
            path=self.path, category="particles", step=first_step
        )
        if self.coordinate_system is None:
            if "Coordinates" not in attributes:
                raise ValueError("Coordinates not found in attributes for particles.")
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
                    "particles": (
                        remap_prtl_quantities_cart
                        if self.coordinate_system == CoordinateSystem.XYZ
                        else remap_prtl_quantities_sph
                    ),
                }
            )

        return (
            quantities,
            sp_with_idx,
            sp_without_idx,
            attributes,
            ParticleDataset(
                species=all_species,
                steps=self.steps,
                times=self.times,
                colnames=[
                    (
                        self.remap["particles"](q)
                        if (self.remap is not None and "particles" in self.remap)
                        else q
                    )
                    for q in unique_quantities
                ]
                + ["id", "sp"],
                read_column=self._read_column,
                partition_lengths=partition_lengths,
            ),
        )

    @property
    def particles_defined(self) -> bool:
        """bool: Whether the particles category is defined."""
        return self.__particles_defined

    @property
    def particles(self) -> Union[ParticleDataset, None]:
        """Returns the particles data.

        Returns
        -------
        ParticleDataset | None
            The particles data if defined, otherwise None.

        """
        return self.__particles

    @property
    def attrs(self) -> Dict[str, Any]:
        """dict: The attributes of the particles dataframe."""
        if self.particles_defined:
            return self.attributes
        else:
            return {}

    def help_particles(self, prepend: str = "") -> str:
        return self.particles.help(prepend) if self.particles is not None else ""

    def _get_count(self, step: int, sp: int) -> np.int64:
        try:
            return np.int64(
                self.reader.ReadArrayShapeAtTimestep(
                    self.path, "particles", f"pX1_{sp}", step
                )[0]
            )
        except Exception:
            return np.int64(0)

    def _species_has_quantity(self, read_colname: str, step: int, sp: int) -> bool:
        return f"{read_colname}_{sp}" in self.reader.ReadCategoryNamesAtTimestep(
            self.path, "particles", "p", step
        )

    def _get_quantity_for_species(
        self,
        read_colname: str,
        step: int,
        sp: int,
    ) -> npt.NDArray[Union[np.float64, np.int64]]:
        if f"{read_colname}_{sp}" in self.quantities:
            return self.reader.ReadArrayAtTimestep(
                self.path, "particles", f"{read_colname}_{sp}", step
            )
        else:
            return np.zeros(self._get_count(step, sp)) * np.nan

    def _read_column(
        self, step: int, colname: str
    ) -> npt.NDArray[Union[np.float64, np.int64, np.float32, np.int32]]:
        read_colname = None
        if colname == "id":
            idx = np.concatenate(
                [
                    self.reader.ReadArrayAtTimestep(
                        self.path, "particles", f"pIDX_{sp}", step
                    ).astype(np.int64)
                    for sp in self.sp_with_idx
                ]
                + [
                    np.zeros(self._get_count(step, sp), dtype=np.int64) - 100
                    for sp in self.sp_without_idx
                ]
            )
            if (
                len(self.sp_with_idx) > 0
                and f"pRNK_{self.sp_with_idx[0]}" in self.quantities
            ):
                rnk = np.concatenate(
                    [
                        self.reader.ReadArrayAtTimestep(
                            self.path, "particles", f"pRNK_{sp}", step
                        ).astype(np.int64)
                        for sp in self.sp_with_idx
                    ]
                    + [
                        np.zeros(self._get_count(step, sp), dtype=np.int64) - 100
                        for sp in self.sp_without_idx
                    ]
                )
                return (idx + rnk) * (idx + rnk + 1) // 2 + rnk
            else:
                return idx
        elif colname == "x" or colname == "r":
            read_colname = "pX1"
        elif colname == "y" or colname == "th":
            read_colname = "pX2"
        elif colname == "z" or colname == "ph":
            read_colname = "pX3"
        elif colname == "ux" or colname == "ur":
            read_colname = "pU1"
        elif colname == "uy" or colname == "uth":
            read_colname = "pU2"
        elif colname == "uz" or colname == "uph":
            read_colname = "pU3"
        elif colname == "w":
            read_colname = "pW"
        elif colname == "sp":
            return np.concatenate(
                [
                    np.zeros(self._get_count(step, sp), dtype=np.int32) + sp
                    for sp in self.sp_with_idx
                ]
                + [
                    np.zeros(self._get_count(step, sp), dtype=np.int32) + sp
                    for sp in self.sp_without_idx
                ]
            )
        else:
            read_colname = f"p{colname}"

        return np.concatenate(
            [
                self._get_quantity_for_species(read_colname, step, sp)
                for sp in self.sp_with_idx
                if self._species_has_quantity(read_colname, step, sp)
            ]
            + [
                self._get_quantity_for_species(read_colname, step, sp)
                for sp in self.sp_without_idx
                if self._species_has_quantity(read_colname, step, sp)
            ]
        )
