# pyright: reportMissingTypeStubs=false

from typing import Any, override
import re
import os
import numpy as np

import adios2 as bp

from nt2.utils import Format, Layout
from nt2.readers.base import BaseReader


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
    ) -> dict[str, np.ndarray]:
        variables: list[str] = []
        for filename in self.GetValidFiles(
            path=path,
            category=category,
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
    def ReadEdgeCoordsAtTimestep(
        self,
        path: str,
        step: int,
    ) -> dict[str, Any]:
        dct: dict[str, Any] = {}
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
    ) -> Any:
        with bp.FileReader(filename := self.FullPath(path, category, step)) as f:
            if quantity in f.available_variables():
                var = f.inquire_variable(quantity)
                if var is not None:
                    return f.read(var)
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
    ) -> tuple[int]:
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
    def ReadFieldCoordsAtTimestep(self, path: str, step: int) -> dict[str, Any]:
        with bp.FileReader(filename := self.FullPath(path, "fields", step)) as f:

            def get_coord(c: str) -> Any:
                f_c = f.inquire_variable(c)
                if f_c is not None:
                    return f.read(f_c)
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
