import os
import h5py
import xarray as xr

from nt2.containers.container import Container
from nt2.containers.utils import (
    _read_category_metadata,
    _read_coordinates,
    _preload_domain_shapes,
    _preload_field,
    _preload_field_with_ghosts,
)

from nt2.plotters.polar import (
    _datasetPolarPlotAccessor,
    _polarPlotAccessor,
)

from nt2.plotters.inspect import _datasetInspectPlotAccessor
from nt2.plotters.movie import _moviePlotAccessor

from nt2.containers.utils import InheritClassDocstring


@xr.register_dataset_accessor("polar")
@InheritClassDocstring
class DatasetPolarPlotAccessor(_datasetPolarPlotAccessor):
    pass


@xr.register_dataarray_accessor("polar")
@InheritClassDocstring
class PolarPlotAccessor(_polarPlotAccessor):
    pass


@xr.register_dataset_accessor("inspect")
@InheritClassDocstring
class DatasetInspectPlotAccessor(_datasetInspectPlotAccessor):
    pass


@xr.register_dataarray_accessor("movie")
@InheritClassDocstring
class MoviePlotAccessor(_moviePlotAccessor):
    pass


class FieldsContainer(Container):
    """
    * * * * FieldsContainer : Container * * * *

    Class for hodling the field (grid-based) data.

    Attributes
    ----------
    fields : xarray.Dataset
        The xarray dataset for all the field quantities.

    fields_files : list
        The list of opened fields files.

    Methods
    -------
    print_fields()
        Prints the basic information about the field data.

    """

    def __init__(self, **kwargs):
        super(FieldsContainer, self).__init__(**kwargs)
        QuantityDict = {
            "Ttt": "E",
            "Ttx": "Px",
            "Tty": "Py",
            "Ttz": "Pz",
        }
        CoordinateDict = {
            "cart": {"x": "x", "y": "y", "z": "z", "1": "x", "2": "y", "3": "z"},
            "sph": {
                "r": "r",
                "theta": "θ" if self.configs["use_greek"] else "th",
                "phi": "φ" if self.configs["use_greek"] else "ph",
                "1": "r",
                "2": "θ" if self.configs["use_greek"] else "th",
                "3": "φ" if self.configs["use_greek"] else "ph",
            },
        }
        if self.configs["single_file"]:
            assert self.master_file is not None, "Master file not found"
            self.metadata["fields"] = _read_category_metadata(
                True, "f", self.master_file
            )
        else:
            field_path = os.path.join(self.path, "fields")
            if os.path.isdir(field_path):
                files = sorted(os.listdir(field_path))
                try:
                    self.fields_files = [
                        h5py.File(os.path.join(field_path, f), "r") for f in files
                    ]
                except OSError:
                    raise OSError(f"Could not open file in {field_path}")
                self.metadata["fields"] = _read_category_metadata(
                    False, "f", self.fields_files
                )

        if not self.isDebug():
            coords = list(CoordinateDict[self.configs["coordinates"]].values())[::-1][
                -self.configs["dimension"] :
            ]
        else:
            coords = ["i3", "i2", "i1"][-self.configs["dimension"] :]

        if self.configs["single_file"]:
            assert self.master_file is not None, "Master file not found"
            self.mesh = _read_coordinates(coords, self.master_file)
        else:
            self.mesh = _read_coordinates(coords, self.fields_files[0])

        self._fields = xr.Dataset()

        if "fields" in self.metadata and len(self.metadata["fields"]["outsteps"]) > 0:
            self.domains = xr.Dataset()
            for i in range(self.configs["dimension"]):
                self.domains[f"x{i+1}"], self.domains[f"sx{i+1}"] = (
                    _preload_domain_shapes(
                        single_file=self.configs["single_file"],
                        k=f"N{i+1}l",
                        outsteps=self.metadata["fields"]["outsteps"],
                        times=self.metadata["fields"]["times"],
                        steps=self.metadata["fields"]["steps"],
                        file=(
                            self.master_file
                            if self.configs["single_file"]
                            and self.master_file is not None
                            else self.fields_files
                        ),
                    )
                )

            for k in self.metadata["fields"]["quantities"]:
                if not self.isDebug():
                    name, dset = _preload_field(
                        single_file=self.configs["single_file"],
                        k=k,
                        outsteps=self.metadata["fields"]["outsteps"],
                        times=self.metadata["fields"]["times"],
                        steps=self.metadata["fields"]["steps"],
                        coords=coords,
                        xc_coords=self.mesh["xc"],
                        xe_min_coords=self.mesh["xe_min"],
                        xe_max_coords=self.mesh["xe_max"],
                        coord_replacements=list(
                            CoordinateDict[self.configs["coordinates"]].items()
                        ),
                        field_replacements=list(QuantityDict.items()),
                        layout=self.configs["layout"],
                        file=(
                            self.master_file
                            if self.configs["single_file"]
                            and self.master_file is not None
                            else self.fields_files
                        ),
                    )
                else:
                    (
                        name,
                        dset,
                        self.mesh["xc"],
                        self.mesh["xe_min"],
                        self.mesh["xe_max"],
                    ) = _preload_field_with_ghosts(
                        single_file=self.configs["single_file"],
                        k=k,
                        outsteps=self.metadata["fields"]["outsteps"],
                        times=self.metadata["fields"]["times"],
                        steps=self.metadata["fields"]["steps"],
                        coords=coords,
                        coord_replacements=list(
                            CoordinateDict[self.configs["coordinates"]].items()
                        ),
                        field_replacements=list(QuantityDict.items()),
                        layout=self.configs["layout"],
                        file=(
                            self.master_file
                            if self.configs["single_file"]
                            and self.master_file is not None
                            else self.fields_files
                        ),
                    )
                self.fields[name] = dset

    @property
    def fields(self):
        return self._fields

    def __del__(self):
        if not self.configs["single_file"]:
            for f in self.fields_files:
                f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def print_fields(self) -> str:
        def sizeof_fmt(num, suffix="B"):
            for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
                if abs(num) < 1e3:
                    return f"{num:3.1f} {unit}{suffix}"
                num /= 1e3
            return f"{num:.1f} Y{suffix}"

        def compactify(lst):
            c = ""
            cntr = 0
            for l_ in lst:
                if cntr > 5:
                    c += "\n                "
                    cntr = 0
                c += l_ + ", "
                cntr += 1
            return c[:-2]

        string = ""
        field_keys = list(self.fields.data_vars.keys())

        if len(field_keys) > 0:
            string += "Fields:\n"
            string += f"  - data axes: {compactify(self.fields.indexes.keys())}\n"
            string += f"  - timesteps: {self.fields[field_keys[0]].shape[0]}\n"
            string += f"  - shape: {self.fields[field_keys[0]].shape[1:]}\n"
            string += f"  - quantities: {compactify(self.fields.data_vars.keys())}\n"
            string += f"  - total size: {sizeof_fmt(self.fields.nbytes)}\n"
        else:
            string += "Fields: empty\n"

        return string

    def plotDomains(self, ax, ti=None, t=None, **kwargs):
        if self.domains is None:
            raise AttributeError("Domains not found")

        assert len(self.domains.data_vars) == 4, "Data must be 2D for plotGrid to work"

        import matplotlib.patches as mpatches

        ngh = self.configs["ngh"]

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        options = {
            "lw": 2,
            "color": "r",
            "ls": "-",
        }
        options.update(kwargs)

        for dom in self.domains.dom:
            selection = self.domains.sel(dom=dom)
            if ti is not None:
                selection = selection.sel(t=ti)
            elif t is not None:
                selection = selection.sel(t=t, method="nearest")
            else:
                selection = selection.isel(t=0)

            x1c, sx1 = selection.x1.values[()], selection.sx1.values[()]
            x2c, sx2 = selection.x2.values[()], selection.sx2.values[()]

            # add rectangle
            ax.add_patch(
                mpatches.Rectangle(
                    (x1c + ngh, x2c + ngh),
                    sx1 - 2 * ngh,
                    sx2 - 2 * ngh,
                    fill=None,
                    **options,
                )
            )

            # ax.plot(
            #     self.domains[x1][j],
            #     self.domains[x2][j],
            #     **options,
            # )
            # ax.plot(
            #     self.domains[x1_e][j],
            #     self.domains[x2_e][j],
            #     **options,
            # )
