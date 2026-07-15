__version__ = "1.6.0"

import xarray as xr
from .containers.data import Data as nt2Data

from .plotters import polar as acc_polar
from .plotters import particles as acc_particles
from .plotters import inspect as acc_inspect
from .plotters import movie as acc_movie
from .utils import InheritClassDocstring


class Data(nt2Data):
    pass


# patch until xarray decides to actually fix this stupid bug
def __patch_xarray_dark(cls):
    STYLE = """
    <style>
    .xr-wrap {
    --xr-background-color-row-odd: rgba(100, 100, 100, 0.10) !important;
    --xr-background-color-row-even: rgba(30, 30, 30, 0.10) !important;
    }
    </style>
    """
    cls._repr_html_orig_ = cls._repr_html_
    cls._repr_html_ = lambda self: STYLE + cls._repr_html_orig_(self)


__patch_xarray_dark(xr.DataArray)
__patch_xarray_dark(xr.Dataset)


# register custom plotters
@xr.register_dataset_accessor("polar")
@InheritClassDocstring
class DatasetPolarPlotAccessor(acc_polar.ds_accessor):
    pass


@xr.register_dataset_accessor("particles")
@InheritClassDocstring
class DatasetParticlesPlotAccessor(acc_particles.ds_accessor):
    pass


@xr.register_dataarray_accessor("polar")
@InheritClassDocstring
class PolarPlotAccessor(acc_polar.accessor):
    pass


@xr.register_dataset_accessor("inspect")
@InheritClassDocstring
class DatasetInspectPlotAccessor(acc_inspect.ds_accessor):
    pass


@xr.register_dataarray_accessor("movie")
@InheritClassDocstring
class MoviePlotAccessor(acc_movie.accessor):
    pass
