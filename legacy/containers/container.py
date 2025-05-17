import os
import h5py
import numpy as np
from typing import Any


def _read_attribs_SingleFile(file: h5py.File):
    attribs = {}
    for k in file.attrs.keys():
        attr = file.attrs[k]
        if type(attr) is bytes or type(attr) is np.bytes_:
            attribs[k] = attr.decode("UTF-8")
        else:
            attribs[k] = attr
    return attribs


class Container:
    """
    * * * * Container * * * *

    Parent class for all data containers.

    Args
    ----
    path : str
        The path to the data.

    Kwargs
    ------
    pickle : bool, optional
        Whether to use pickle for reading the data. Default is True.

    greek : bool, optional
        Whether to use Greek letters for the spherical coordinates. Default is False.

    dask_props : dict, optional
        Additional properties for Dask [NOT IMPLEMENTED]. Default is {}.

    Attributes
    ----------
    path : str
        The path to the data.

    configs : dict
        The configuration settings for the data.

    metadata : dict
        The metadata for the data.

    mesh : dict
        Coordinate grid of the domain (cell-centered & edges).

    master_file : h5py.File
        The master file for the data (from which the main attributes are read).

    attrs : dict
        The attributes of the master file.

    Methods
    -------
    plotGrid(ax, **kwargs)
        Plots the gridlines of the domain.

    """

    def __init__(self, path, pickle=True, greek=False, dask_props={}):
        super(Container, self).__init__()

        self.path = path
        self.configs: dict[str, Any] = {
            "single_file": self.path.endswith(".h5"),
            "use_pickle": pickle,
            "use_greek": greek,
        }
        self.metadata = {}
        self.mesh = None
        if self.configs["single_file"]:
            try:
                self.master_file: h5py.File | None = h5py.File(self.path, "r")
            except OSError:
                raise OSError(f"Could not open file {self.path}")
        else:
            field_path = os.path.join(self.path, "fields")
            file = os.path.join(field_path, os.listdir(field_path)[0])
            try:
                self.master_file: h5py.File | None = h5py.File(file, "r")
            except OSError:
                raise OSError(f"Could not open file {file}")

        self.attrs = _read_attribs_SingleFile(self.master_file)

        assert self.master_file is not None, "Master file not found"

        self.configs["ngh"] = int(self.master_file.attrs.get("NGhosts", 0))
        self.configs["layout"] = (
            "right" if self.master_file.attrs.get("LayoutRight", 1) == 1 else "left"
        )
        self.configs["dimension"] = int(self.master_file.attrs.get("Dimension", 1))
        self.configs["coordinates"] = self.master_file.attrs.get(
            "Coordinates", b"cart"
        ).decode("UTF-8")
        if self.configs["coordinates"] == "qsph":
            self.configs["coordinates"] = "sph"

        if self.isDebug():
            self.configs["coordinates"] = "cart"

        if not self.configs["single_file"]:
            self.master_file.close()
            self.master_file = None

    def isDebug(self):
        return self.configs["ngh"] > 0

    def __del__(self):
        if self.master_file is not None:
            self.master_file.close()

    def plotGrid(self, ax, **kwargs):
        try:
            assert self.mesh is not None, "Mesh not found"
        except AttributeError:
            raise AttributeError("Mesh not found")
        except AssertionError:
            raise AssertionError("Mesh not found")

        assert len(self.mesh["xc"]) == 2, "Data must be 2D for plotGrid to work"

        from matplotlib import patches as mpatches

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        options = {
            "lw": 1,
            "color": "k",
            "ls": "-",
        }
        options.update(kwargs)

        x1_emin, x2_emin = list(self.mesh["xe_min"].keys())
        x1_emax, x2_emax = list(self.mesh["xe_max"].keys())
        x1_e = list(self.mesh["xe_min"][x1_emin][1]) + [
            self.mesh["xe_max"][x1_emax][1][-1]
        ]
        x2_e = list(self.mesh["xe_min"][x2_emin][1]) + [
            self.mesh["xe_max"][x2_emax][1][-1]
        ]
        if self.configs["coordinates"] == "cart":
            for x1 in x1_e:
                ax.plot([x1, x1], [x2_e[0], x2_e[-1]], **options)
            for x2 in x2_e:
                ax.plot([x1_e[0], x1_e[-1]], [x2, x2], **options)
        else:
            for r in x1_e:
                ax.add_patch(
                    mpatches.Arc(
                        (0, 0),
                        2 * r,
                        2 * r,
                        theta1=-90,
                        theta2=90,
                        fill=False,
                        **options,
                    )
                )
            for th in x2_e:
                ax.plot(
                    [
                        x1_e[0] * np.sin(th),
                        x1_e[-1] * np.sin(th),
                    ],
                    [
                        x1_e[0] * np.cos(th),
                        x1_e[-1] * np.cos(th),
                    ],
                    **options,
                )
        ax.set(xlim=xlim, ylim=ylim)
