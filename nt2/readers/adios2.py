from __future__ import annotations

import sys
from typing import Any

from tqdm import tqdm

if sys.version_info >= (3, 12):
    from typing import override
else:

    def override(method):
        return method


import os
import re

import adios2 as bp
import numpy as np
import numpy.typing as npt

from nt2.readers.base import BaseReader
from nt2.utils import Format, Layout


class Reader(BaseReader):
    @property
    @override
    def format(self) -> Format:
        return Format.BP5

    @staticmethod
    @override
    def EnterFile(
        filename: str,
    ) -> bp.FileReader:
        return bp.FileReader(filename)

    @override
    def ReadPerTimestepVariable(
        self,
        path: str,
        category: str,
        varname: str,
        newname: str,
        valid_files: list[str],
    ) -> dict[str, npt.NDArray[Any]]:
        variables: list[float] = []
        for filename in tqdm(
            valid_files,
            desc=f"Reading {category} {varname}",
            position=0,
            leave=False,
        ):
            with bp.FileReader(os.path.join(path, category, filename)) as f:
                avail: dict[str, Any] = f.available_variables()
                vars: list[str] = list(avail.keys())
                if varname in vars:
                    var = f.inquire_variable(varname)
                    if var is not None:
                        variables.append(f.read(var))
                    else:
                        raise ValueError(
                            f"{varname} is not a variable in the BP file {filename}"
                        )
                else:
                    raise ValueError(f"{varname} not found in the BP file {filename}")
        return {newname: np.array(variables)}

    @override
    def ReadPerTimestepVariables(
        self,
        path: str,
        category: str,
        varnames: list[str],
        newnames: list[str],
        valid_files: list[str],
    ) -> dict[str, npt.NDArray[Any]]:
        variables = {newname: [] for newname in newnames}
        for filename in tqdm(
            valid_files,
            desc=f"Reading {category} {varnames}",
            position=0,
            leave=False,
        ):
            with bp.FileReader(os.path.join(path, category, filename)) as f:
                avail: dict[str, Any] = f.available_variables()
                vars: list[str] = list(avail.keys())
                for varname, newname in zip(varnames, newnames):
                    if varname in vars:
                        var = f.inquire_variable(varname)
                        if var is not None:
                            variables[newname].append(f.read(var))
                        else:
                            raise ValueError(
                                f"{varname} is not a variable in the BP file {filename}"
                            )
                    else:
                        raise ValueError(
                            f"{varname} not found in the BP file {filename}"
                        )
        return {newname: np.array(variables[newname]) for newname in newnames}

    @override
    def ReadParticleCountsAtTimestep(
        self, path: str, step: int, species: list[int]
    ) -> dict[int, int]:
        """Read all per-species counts from one BP file's metadata."""
        with bp.FileReader(self.FullPath(path, "particles", step)) as f:
            available = f.available_variables()
            counts: dict[int, int] = {}
            for sp in species:
                name = f"pX1_{sp}"
                var = f.inquire_variable(name) if name in available else None
                shape = var.shape() if var is not None else []
                counts[sp] = int(shape[0]) if shape else 0
            return counts

    @override
    def ReadEdgeCoordsAtTimestep(
        self,
        path: str,
        step: int,
    ) -> dict[str, Any]:
        dct: dict[str, npt.NDArray[Any]] = {}
        with bp.FileReader(self.FullPath(path, "fields", step)) as f:
            avail: dict[str, Any] = f.available_variables()
            vars: list[str] = list(avail.keys())
            for var in vars:
                if var.startswith("X") and var.endswith("e"):
                    var_obj = f.inquire_variable(var)
                    if var_obj is not None:
                        dct[var] = f.read(var_obj)
        return dct

    @override
    def ReadAttrsAtTimestep(
        self,
        path: str,
        category: str,
        step: int,
    ) -> dict[str, Any]:
        with bp.FileReader(self.FullPath(path, category, step)) as f:
            return {k: f.read_attribute(k) for k in f.available_attributes()}

    @override
    def ReadArrayAtTimestep(
        self,
        path: str,
        category: str,
        quantity: str,
        step: int,
    ) -> npt.NDArray[Any]:
        with bp.FileReader(filename := self.FullPath(path, category, step)) as f:
            if quantity in f.available_variables():
                var = f.inquire_variable(quantity)
                if var is not None:
                    if var.shape() == [0]:
                        return np.array([])
                    else:
                        return np.array(f.read(var))
                else:
                    raise ValueError(f"{quantity} not found in the {filename}")
            else:
                raise ValueError(f"{quantity} not found in the {filename}")

    @override
    def ReadCategoryNamesAtTimestep(
        self,
        path: str,
        category: str,
        prefix: str,
        step: int,
    ) -> set[str]:
        with bp.FileReader(self.FullPath(path, category, step)) as f:
            keys: list[str] = f.available_variables()
            return set(
                filter(
                    lambda c: c.startswith(prefix),
                    keys,
                )
            )

    @override
    def ReadArrayShapeAtTimestep(
        self, path: str, category: str, quantity: str, step: int
    ) -> tuple[int, ...]:
        with bp.FileReader(filename := self.FullPath(path, category, step)) as f:
            if quantity in f.available_variables():
                var = f.inquire_variable(quantity)
                if var is not None:
                    return var.shape()
                else:
                    raise ValueError(
                        f"{category.capitalize()} {quantity} is not a group in the {filename}"
                    )
            else:
                raise ValueError(
                    f"{category.capitalize()} {quantity} not found in the {filename}"
                )

    @override
    def ReadArrayShapeExplicitlyAtTimestep(
        self, path: str, category: str, quantity: str, step: int
    ) -> tuple[int, ...]:
        with bp.FileReader(filename := self.FullPath(path, category, step)) as f:
            if quantity in f.available_variables():
                var = f.inquire_variable(quantity)
                if var is not None and (read := f.read(var)) is not None:
                    return read.shape
                else:
                    raise ValueError(
                        f"{category.capitalize()} {quantity} is not a group in the {filename}"
                    )
            else:
                raise ValueError(
                    f"{category.capitalize()} {quantity} not found in the {filename}"
                )

    @override
    def ReadFieldCoordsAtTimestep(
        self, path: str, step: int
    ) -> dict[str, npt.NDArray[Any]]:
        with bp.FileReader(filename := self.FullPath(path, "fields", step)) as f:

            def get_coord(c: str) -> npt.NDArray[Any]:
                f_c = f.inquire_variable(c)
                if f_c is not None:
                    return np.array(f.read(f_c))
                else:
                    raise ValueError(f"Field {c} is not a group in the {filename}")

            keys: list[str] = list(f.available_variables())
            return {c: get_coord(c) for c in keys if re.match(r"^X[1|2|3]$", c)}

    @override
    def ReadFieldLayoutAtTimestep(self, path: str, step: int) -> Layout:
        with bp.FileReader(filename := self.FullPath(path, "fields", step)) as f:
            attrs: dict[str, Any] = f.available_attributes()
            keys = list(attrs.keys())
            if "LayoutRight" not in keys:
                raise ValueError(f"LayoutRight attribute not found in the {filename}")
            return Layout.R if f.read_attribute("LayoutRight") else Layout.L
