from typing import Any
import re
import os
import numpy as np

import adios2 as bp

from nt2.utils import Format, Layout
from nt2.readers.base import BaseReader


class Reader(BaseReader):
    @property
    def format(self) -> Format:
        return Format.BP5

    @staticmethod
    def EnterFile(
        filename: str,
    ) -> bp.FileReader:
        return bp.FileReader(filename)

    def ReadPerTimestepVariable(
        self,
        path: str,
        category: str,
        varname: str,
        newname: str,
    ) -> dict[str, np.ndarray]:
        variables = []
        for filename in self.GetValidFiles(
            path=path,
            category=category,
        ):
            with bp.FileReader(os.path.join(path, category, filename)) as f:
                vars = list(f.available_variables().keys())
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

    def ReadAttrsAtTimestep(
        self,
        path: str,
        category: str,
        step: int,
    ) -> dict[str, Any]:
        with bp.FileReader(self.FullPath(path, category, step)) as f:
            return {k: f.read_attribute(k) for k in f.available_attributes()}

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

    def ReadCategoryNamesAtTimestep(
        self,
        path: str,
        category: str,
        prefix: str,
        step: int,
    ) -> set[str]:
        with bp.FileReader(self.FullPath(path, category, step)) as f:
            return set(c for c in f.available_variables() if c.startswith(prefix))

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

    def ReadFieldCoordsAtTimestep(self, path: str, step: int) -> dict[str, Any]:
        with bp.FileReader(filename := self.FullPath(path, "fields", step)) as f:

            def get_coord(c):
                f_c = f.inquire_variable(c)
                if f_c is not None:
                    return f.read(f_c)
                else:
                    raise ValueError(f"Field {c} is not a group in the {filename}")

            return {
                c: get_coord(c)
                for c in list(f.available_variables())
                if re.match(r"^X[1|2|3]$", c)
            }

    def ReadFieldLayoutAtTimestep(self, path: str, step: int) -> Layout:
        with bp.FileReader(filename := self.FullPath(path, "fields", step)) as f:
            if "LayoutRight" not in list(f.available_attributes().keys()):
                raise ValueError(f"LayoutRight attribute not found in the {filename}")
            return Layout.R if f.read_attribute("LayoutRight") else Layout.L
