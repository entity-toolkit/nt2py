import os
import h5py
import numpy as np
from typing import Any
from dask.distributed import Client


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
    def __init__(
        self, path, single_file=False, pickle=True, greek=False, dask_props={}
    ):
        super(Container, self).__init__()

        self.client = Client(**dask_props)
        if self.client.status == "running":
            print("Dask client launched:")
            print(self.client)

        self.configs: dict[str, Any] = {
            "single_file": single_file,
            "use_pickle": pickle,
            "use_greek": greek,
        }
        self.path = path
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

        if not self.configs["single_file"]:
            self.master_file.close()
            self.master_file = None
            # if coordinates == "sph":
            #     self.metric = SphericalMetric()
            # else:
            #     self.metric = MinkowskiMetric()

    def __del__(self):
        if self.master_file is not None:
            self.master_file.close()
        if self.client.status == "running":
            self.client.close()

    def plotGrid(self, ax, **kwargs):
        from matplotlib import patches

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        options = {
            "lw": 1,
            "color": "k",
            "ls": "-",
        }
        options.update(kwargs)

        if self.configs["coordinates"] == "cart":
            for x in self.attrs["X1"]:
                ax.plot([x, x], [self.attrs["X2Min"], self.attrs["X2Max"]], **options)
            for y in self.attrs["X2"]:
                ax.plot([self.attrs["X1Min"], self.attrs["X1Max"]], [y, y], **options)
        else:
            for r in self.attrs["X1"]:
                ax.add_patch(
                    patches.Arc(
                        (0, 0),
                        2 * r,
                        2 * r,
                        theta1=-90,
                        theta2=90,
                        fill=False,
                        **options,
                    )
                )
            for th in self.attrs["X2"]:
                ax.plot(
                    [
                        self.attrs["X1Min"] * np.sin(th),
                        self.attrs["X1Max"] * np.sin(th),
                    ],
                    [
                        self.attrs["X1Min"] * np.cos(th),
                        self.attrs["X1Max"] * np.cos(th),
                    ],
                    **options,
                )
        ax.set(xlim=xlim, ylim=ylim)

    def print_container(self) -> str:
        return f"Client {self.client}\n"

    # def makeMovie(self, plot, makeframes=True, **kwargs):
    #     """
    #     Makes a movie from a plot function
    #
    #     Parameters
    #     ----------
    #     plot : function
    #         The plot function to use; accepts output timestep and dataset as arguments.
    #     makeframes : bool, optional
    #         Whether to make the frames, or just proceed to making the movie. Default is True.
    #     num_cpus : int, optional
    #         The number of CPUs to use for making the frames. Default is None.
    #     **kwargs :
    #         Additional keyword arguments passed to `ffmpeg`.
    #     """
    #     import numpy as np
    #
    #     if makeframes:
    #         makemovie = all(
    #             exp.makeFrames(
    #                 plot,
    #                 np.arange(len(self.t)),
    #                 f"{self.attrs['simulation.name']}/frames",
    #                 data=self,
    #                 num_cpus=kwargs.pop("num_cpus", None),
    #             )
    #         )
    #     else:
    #         makemovie = True
    #     if makemovie:
    #         exp.makeMovie(
    #             input=f"{self.attrs['simulation.name']}/frames/",
    #             overwrite=True,
    #             output=f"{self.attrs['simulation.name']}.mp4",
    #             number=5,
    #             **kwargs,
    #         )
    #         return True
