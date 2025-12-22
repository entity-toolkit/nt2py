from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sys

if sys.version_info >= (3, 12):
    from typing import override
else:

    def override(method):
        return method


import re
import os
import numpy as np
import numpy.typing as npt

try:
    import h5py  # runtime
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import h5py as _h5py

from nt2.utils import Format, Layout
from nt2.readers.base import BaseReader


def _require_h5py():
    if h5py is None:
        raise ImportError(
            "HDF5 support requires the optional dependency 'h5py'. "
            "Install it with: pip install nt2py[hdf5]"
        )
    return h5py


class Reader(BaseReader):
    @staticmethod
    def __extract_step0(f: "_h5py.File") -> "_h5py.Group":
        h5 = _require_h5py()
        if "Step0" in f.keys():
            f0 = f["Step0"]
            if isinstance(f0, h5.Group):
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
    ) -> "_h5py.File":
        h5 = _require_h5py()
        return h5.File(filename, "r")

    @override
    def ReadPerTimestepVariable(
        self,
        path: str,
        category: str,
        varname: str,
        newname: str,
    ) -> dict[str, npt.NDArray[Any]]:
        variables: list[Any] = []
        h5 = _require_h5py()
        for filename in self.GetValidFiles(
            path=path,
            category=category,
        ):
            with h5.File(os.path.join(path, category, filename), "r") as f:
                f0 = Reader.__extract_step0(f)
                if varname in f0.keys():
                    var = f0[varname]
                    if isinstance(var, h5.Dataset):
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
        h5 = _require_h5py()
        with h5.File(self.FullPath(path, category, step), "r") as f:
            return {k: v for k, v in f.attrs.items()}

    @override
    def ReadEdgeCoordsAtTimestep(
        self,
        path: str,
        step: int,
    ) -> dict[str, npt.NDArray[Any]]:
        h5 = _require_h5py()
        with h5.File(self.FullPath(path, "fields", step), "r") as f:
            f0 = Reader.__extract_step0(f)
            return {k: v[:] for k, v in f0.items() if k[0] == "X" and k[-1] == "e"}

    @override
    def ReadArrayAtTimestep(
        self,
        path: str,
        category: str,
        quantity: str,
        step: int,
    ) -> npt.NDArray[Any]:
        h5 = _require_h5py()
        with h5.File(filename := self.FullPath(path, category, step), "r") as f:
            f0 = Reader.__extract_step0(f)
            if quantity in f0.keys():
                var = f0[quantity]
                if isinstance(var, h5.Dataset):
                    return np.array(var[:])
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
        h5 = _require_h5py()
        with h5.File(self.FullPath(path, category, step), "r") as f:
            f0 = Reader.__extract_step0(f)
            keys: list[str] = list(f0.keys())
            return set(c for c in keys if c.startswith(prefix))

    @override
    def ReadArrayShapeAtTimestep(
        self, path: str, category: str, quantity: str, step: int
    ) -> tuple[int]:
        h5 = _require_h5py()
        with h5.File(filename := self.FullPath(path, category, step), "r") as f:
            f0 = Reader.__extract_step0(f)
            if quantity in f0.keys():
                var = f0[quantity]
                if isinstance(var, h5.Dataset):
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
    def ReadArrayShapeExplicitlyAtTimestep(
        self, path: str, category: str, quantity: str, step: int
    ) -> tuple[int]:
        h5 = _require_h5py()
        with h5.File(self.FullPath(path, category, step), "r") as f:
            f0 = Reader.__extract_step0(f)
            if quantity in f0.keys():
                var = f0[quantity]
                if isinstance(var, h5.Dataset) and (read := var[:]) is not None:
                    return read.shape
                else:
                    raise ValueError(
                        f"{category.capitalize()} {quantity} is not a group in the HDF5 file"
                    )
            else:
                raise ValueError(
                    f"{category.capitalize()} {quantity} not found in the HDF5 file"
                )

    @override
    def ReadFieldCoordsAtTimestep(
        self, path: str, step: int
    ) -> dict[str, npt.NDArray[Any]]:
        h5 = _require_h5py()
        with h5.File(filename := self.FullPath(path, "fields", step), "r") as f:
            f0 = Reader.__extract_step0(f)

            def get_coord(c: str) -> Any:
                f0_c = f0[c]
                if isinstance(f0_c, h5.Dataset):
                    return f0_c[:]
                else:
                    raise ValueError(f"Field {c} is not a group in the {filename}")

            keys: list[str] = list(f0.keys())
            return {c: get_coord(c) for c in keys if re.match(r"^X[1|2|3]$", c)}

    @override
    def ReadFieldLayoutAtTimestep(self, path: str, step: int) -> Layout:
        h5 = _require_h5py()
        with h5.File(filename := self.FullPath(path, "fields", step), "r") as f:
            if "LayoutRight" not in f.attrs:
                raise ValueError(f"LayoutRight attribute not found in the {filename}")
            return Layout.R if f.attrs["LayoutRight"] else Layout.L
