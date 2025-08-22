# pyright: reportMissingTypeStubs=false

from typing import Any, override
import re
import os
import numpy as np

import h5py

from nt2.utils import Format, Layout
from nt2.readers.base import BaseReader


class Reader(BaseReader):
    @staticmethod
    def __extract_step0(f: h5py.File) -> h5py.Group:
        if "Step0" in f.keys():
            f0 = f["Step0"]
            if isinstance(f0, h5py.Group):
                return f0
            else:
                raise ValueError(f"Step0 is not a group in the HDF5 file")
        else:
            raise ValueError(f"Wrong structure of the hdf5 file")

    @property
    @override
    def format(self) -> Format:
        return Format.HDF5

    @staticmethod
    @override
    def EnterFile(
        filename: str,
    ) -> h5py.File:
        return h5py.File(filename, "r")

    @override
    def ReadPerTimestepVariable(
        self,
        path: str,
        category: str,
        varname: str,
        newname: str,
    ) -> dict[str, np.ndarray]:
        variables: list[Any] = []
        for filename in self.GetValidFiles(
            path=path,
            category=category,
        ):
            with h5py.File(os.path.join(path, category, filename), "r") as f:
                f0 = Reader.__extract_step0(f)
                if varname in f0.keys():
                    var = f0[varname]
                    if isinstance(var, h5py.Dataset):
                        variables.append(var[()])
                    else:
                        raise ValueError(
                            f"{varname} is not a group in the HDF5 file {filename}"
                        )
                else:
                    raise ValueError(f"{varname} not found in the HDF5 file {filename}")

        return {newname: np.array(variables)}

    @override
    def ReadAttrsAtTimestep(
        self,
        path: str,
        category: str,
        step: int,
    ) -> dict[str, Any]:
        with h5py.File(self.FullPath(path, category, step), "r") as f:
            return {k: v for k, v in f.attrs.items()}

    @override
    def ReadEdgeCoordsAtTimestep(
        self,
        path: str,
        step: int,
    ) -> dict[str, Any]:
        with h5py.File(self.FullPath(path, "fields", step), "r") as f:
            f0 = Reader.__extract_step0(f)
            return {k: v[:] for k, v in f0.items() if k[0] == "X" and k[-1] == "e"}

    @override
    def ReadArrayAtTimestep(
        self,
        path: str,
        category: str,
        quantity: str,
        step: int,
    ) -> Any:
        with h5py.File(filename := self.FullPath(path, category, step), "r") as f:
            f0 = Reader.__extract_step0(f)
            if quantity in f0.keys():
                var = f0[quantity]
                if isinstance(var, h5py.Dataset):
                    return var[:]
                else:
                    raise ValueError(f"{quantity} is not a group in the {filename}")
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
        with h5py.File(self.FullPath(path, category, step), "r") as f:
            f0 = Reader.__extract_step0(f)
            keys: list[str] = list(f0.keys())
            return set(c for c in keys if c.startswith(prefix))

    @override
    def ReadArrayShapeAtTimestep(
        self, path: str, category: str, quantity: str, step: int
    ) -> tuple[int]:
        with h5py.File(filename := self.FullPath(path, category, step), "r") as f:
            f0 = Reader.__extract_step0(f)
            if quantity in f0.keys():
                var = f0[quantity]
                if isinstance(var, h5py.Dataset):
                    return var.shape
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
        with h5py.File(filename := self.FullPath(path, "fields", step), "r") as f:
            f0 = Reader.__extract_step0(f)

            def get_coord(c: str) -> Any:
                f0_c = f0[c]
                if isinstance(f0_c, h5py.Dataset):
                    return f0_c[:]
                else:
                    raise ValueError(f"Field {c} is not a group in the {filename}")

            keys: list[str] = list(f0.keys())
            return {c: get_coord(c) for c in keys if re.match(r"^X[1|2|3]$", c)}

    @override
    def ReadFieldLayoutAtTimestep(self, path: str, step: int) -> Layout:
        with h5py.File(filename := self.FullPath(path, "fields", step), "r") as f:
            if "LayoutRight" not in f.attrs:
                raise ValueError(f"LayoutRight attribute not found in the {filename}")
            return Layout.R if f.attrs["LayoutRight"] else Layout.L
