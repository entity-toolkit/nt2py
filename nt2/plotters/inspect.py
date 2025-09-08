# pyright: reportMissingTypeStubs=false

from typing import Any, Callable
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import xarray as xr
from nt2.utils import DataIs2DPolar
from nt2.plotters.export import makeFramesAndMovie


class ds_accessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj: xr.Dataset = xarray_obj

    def __axes_grid(
        self,
        grouped_fields: dict[str, list[str]],
        makeplot: Callable,  # pyright: ignore[reportUnknownParameterType,reportMissingTypeArgument]
        nrows: int,
        ncols: int,
        nfields: int,
        size: float,
        aspect: float,
        pad: float,
        **fig_kwargs: Any,
    ) -> tuple[mfigure.Figure, list[plt.Axes]]:
        if aspect > 1:
            axw = size / aspect
            axh = size
        else:
            axw = size
            axh = size * aspect

        fig_w = ncols * (axw + pad) + pad
        fig_h = nrows * axh + (nrows + 1) * pad
        fig = plt.figure(figsize=(fig_w, fig_h), **fig_kwargs)

        gs = fig.add_gridspec(nrows, ncols, wspace=pad / axw, hspace=pad / axh)
        axes = [
            fig.add_subplot(gs[i, j])
            for i in range(nrows)
            for j in range(ncols)
            if (i * ncols + j) < nfields
        ]
        for ax, (g, fields) in zip(axes, grouped_fields.items()):
            for field in fields:
                makeplot(ax, field)
            _ = ax.set_ylabel(g)
            _ = ax.set_title(None)

        return fig, axes

    @staticmethod
    def _fixed_axes_grid_with_cbars(
        fields: list[str],
        makeplot: Callable,  # pyright: ignore[reportUnknownParameterType,reportMissingTypeArgument]
        makecbar: Callable,  # pyright: ignore[reportUnknownParameterType,reportMissingTypeArgument]
        nrows: int,
        ncols: int,
        nfields: int,
        size: float,
        aspect: float,
        pad: float,
        cbar_w: float,
        **fig_kwargs: Any,
    ) -> tuple[mfigure.Figure, list[plt.Axes]]:
        from mpl_toolkits.axes_grid1 import Divider, Size

        if aspect > 1:
            axw = size / aspect
            axh = size
        else:
            axw = size
            axh = size * aspect

        fig_w = ncols * (axw + cbar_w + pad) + pad
        fig_h = nrows * axh + (nrows + 1) * pad
        fig = plt.figure(figsize=(fig_w, fig_h), **fig_kwargs)

        h = []
        for _ in range(ncols):
            h += [Size.Fixed(pad), Size.Fixed(axw), Size.Fixed(cbar_w)]
        h += [Size.Fixed(pad)]

        v = []
        for _ in range(nrows):
            v += [Size.Fixed(pad), Size.Fixed(axh)]
        v += [Size.Fixed(pad)]

        divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
        axes: list[plt.Axes] = []

        cntr = 0
        for i in range(nrows):
            for j in range(ncols):
                cntr += 1
                if cntr > nfields:
                    break
                nx = 3 * j + 1
                ny = 2 * (nrows - 1 - i) + 1

                ax = fig.add_axes(
                    divider.get_position(),
                    axes_locator=divider.new_locator(nx=nx, ny=ny),
                )
                field = fields[cntr - 1]
                im = makeplot(ax, field)
                cax = fig.add_axes(
                    divider.get_position(),
                    axes_locator=divider.new_locator(nx=nx + 1, ny=ny),
                )
                _ = fig.colorbar(im, cax=cax)
                makecbar(ax, cax, field)
                axes.append(ax)
        return fig, axes

    def plot(
        self,
        fig: mfigure.Figure | None = None,
        name: str | None = None,
        skip_fields: list[str] | None = None,
        only_fields: list[str] | None = None,
        fig_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        movie_kwargs: dict[str, Any] | None = None,
        set_aspect: str | None = "equal",
    ) -> mfigure.Figure | bool:
        """
        Plots the overview plot for fields at a given time or step (or as a movie).

        Kwargs
        ------
        fig : matplotlib.figure.Figure | None, optional
            The figure to plot the data (if None, a new figure is created). Default is None.

        name : string | None, optional
            Used when saving the frames and the movie. Default is None.

        skip_fields : list, optional
            The list of fields to skip in the plotting (can contain regex). Default is [].

        only_fields : list, optional
            The list of fields to plot (con contain regex). Default is [].
            If empty, all fields are plotted unless contained in skip_fields).
            Overrides skip_fields.

        fig_kwargs : dict, optional
            Additional keyword arguments for plt.figure. Default is { dpi: 200 }.

        plot_kwargs : dict, optional
            Keyword arguments for each plot. Default is {}.
            Key is a regex pattern to match the field name. Value is dict of kwargs.

        movie_kwargs : dict, optional
            Additional keyword arguments for makeMovie. Default is {}.

        set_aspect : str | None, optional
            If None, the aspect ratio will not be enforced. Otherwise, this value is passed to `set_aspect` method of the axes. Default is 'equal'.

        Returns
        -------
        figure : matplotlib.figure.Figure | boolean
            The figure with the plotted data (if single timestep) or True/False.

        """
        if skip_fields is None:
            skip_fields = []
        if only_fields is None:
            only_fields = []
        if fig_kwargs is None:
            fig_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        if movie_kwargs is None:
            movie_kwargs = {}
        if "t" in self._obj.dims:
            if name is None:
                raise ValueError(
                    "Please provide a name for saving the frames and movie"
                )

            def plot_func(ti: int, _):
                if len(self._obj.dims) == 1:
                    _ = self.plot_frame_1d(
                        self._obj.isel(t=ti),
                        None,
                        skip_fields,
                        only_fields,
                        fig_kwargs,
                        plot_kwargs,
                    )
                elif len(self._obj.dims) == 2:
                    _ = self.plot_frame_2d(
                        self._obj.isel(t=ti),
                        None,
                        skip_fields,
                        only_fields,
                        fig_kwargs,
                        plot_kwargs,
                        set_aspect,
                    )
                else:
                    raise ValueError(
                        "Data has more than 2 dimensions; use .sel or .isel to reduce dimension."
                    )

            return makeFramesAndMovie(
                name=name,
                data=self._obj,
                plot=plot_func,
                times=list(range(len(self._obj.t))),
                **movie_kwargs,
            )
        else:
            if len(self._obj.dims) == 1:
                return self.plot_frame_1d(
                    self._obj,
                    fig,
                    skip_fields,
                    only_fields,
                    fig_kwargs,
                    plot_kwargs,
                )
            elif len(self._obj.dims) == 2:
                return self.plot_frame_2d(
                    self._obj,
                    fig,
                    skip_fields,
                    only_fields,
                    fig_kwargs,
                    plot_kwargs,
                    set_aspect,
                )
            else:
                raise ValueError(
                    "Data has more than 2 dimensions; use .sel or .isel to reduce dimension."
                )

    @staticmethod
    def _get_fields_to_plot(
        data: xr.Dataset, skip_fields: list[str], only_fields: list[str]
    ) -> list[str]:
        import re

        nfields = len(data.data_vars)
        if nfields > 0:
            keys: list[str] = [str(k) for k in data.keys()]
            if len(only_fields) == 0:
                fields_to_plot = [
                    f for f in keys if not any([re.match(sf, f) for sf in skip_fields])
                ]
            else:
                fields_to_plot = [
                    f for f in keys if any([re.match(sf, f) for sf in only_fields])
                ]
        else:
            fields_to_plot = []

        if fields_to_plot == []:
            raise ValueError("No fields to plot.")

        fields_to_plot = sorted(fields_to_plot)
        return fields_to_plot

    @staticmethod
    def _get_fields_minmax(
        data: xr.Dataset, fields: list[str]
    ) -> dict[str, None | tuple[float, float]]:
        minmax: dict[str, None | tuple[float, float]] = {
            "E": None,
            "B": None,
            "J": None,
            "N": None,
            "T": None,
        }
        for fld in fields:
            vmin, vmax = (
                data[fld].min().values[()],
                data[fld].max().values[()],
            )
            if fld[0] in "EBJNT":
                mm = minmax[fld[0]]
                if mm is None:
                    minmax[fld[0]] = (vmin, vmax)
                else:
                    minmax[fld[0]] = (
                        min(mm[0], vmin),
                        max(mm[1], vmax),
                    )
        for f, vv in minmax.items():
            if vv is not None:
                (vmin, vmax) = vv
                if vmin < 0 or f in "EBJ":
                    if abs(vmin) > vmax:
                        vmax = abs(vmin)
                    else:
                        vmin = -vmax
                    minmax[f] = (vmin, vmax)

        return minmax

    def plot_frame_1d(
        self,
        data: xr.Dataset,
        fig: mfigure.Figure | None,
        skip_fields: list[str],
        only_fields: list[str],
        fig_kwargs: dict[str, Any],
        plot_kwargs: dict[str, Any],
    ) -> mfigure.Figure:
        if len(data.dims) != 1:
            raise ValueError("Pass 1D data; use .sel or .isel to reduce dimension.")

        import math, re

        # count the number of subplots
        fields_to_plot = self._get_fields_to_plot(data, skip_fields, only_fields)

        # group fields by their first letter
        grouped_fields: dict[str, list[str]] = {}
        for f in fields_to_plot:
            key = f[0]
            if key not in grouped_fields:
                grouped_fields[key] = []
            grouped_fields[key].append(f)

        nplots = len(grouped_fields)

        aspect = 0.5
        ncols = max(1, int(math.floor(nplots * 1.5 * aspect / (1 + 1.5 * aspect))))
        nrows = max(1, int(math.ceil(nplots / ncols)))

        figsize0 = 3.0

        minmax = self._get_fields_minmax(data, fields_to_plot)
        kwargs = {}
        for fld in fields_to_plot:
            kwargs[fld] = {}
            for fld_kwargs in plot_kwargs:
                if re.match(fld_kwargs, fld):
                    kwargs[fld] = {**plot_kwargs[fld_kwargs]}
                    break

        def make_plot(ax: plt.Axes, fld: str):
            data[fld].plot(ax=ax, label=fld, **kwargs[fld])
            _ = ax.set(ylim=minmax[fld[0]])

        fig, axes = self.__axes_grid(
            grouped_fields=grouped_fields,
            makeplot=make_plot,
            nrows=nrows,
            ncols=ncols,
            nfields=nplots,
            size=figsize0,
            aspect=aspect,
            pad=0.5,
            **fig_kwargs,
        )
        for n, ax in enumerate(axes):
            i = n // ncols
            j = n % ncols

            if j != 0:
                _ = ax.set(
                    ylabel=None,
                    yticklabels=[],
                )
            if (nplots - i * ncols - j) > ncols:
                _ = ax.set(
                    xlabel=None,
                    xticklabels=[],
                )
            _ = ax.legend(loc="best", fontsize="small")
        _ = fig.suptitle(f"t = {data.t.values[()]:.2f}", y=0.95)
        return fig

    def plot_frame_2d(
        self,
        data: xr.Dataset,
        fig: mfigure.Figure | None,
        skip_fields: list[str],
        only_fields: list[str],
        fig_kwargs: dict[str, Any],
        plot_kwargs: dict[str, Any],
        set_aspect: str | None,
    ) -> mfigure.Figure:
        if len(data.dims) != 2:
            raise ValueError("Pass 2D data; use .sel or .isel to reduce dimension.")

        x1, x2 = data.dims

        import matplotlib.colors as mcolors
        import numpy as np
        import math, re

        # count the number of subplots
        fields_to_plot = self._get_fields_to_plot(data, skip_fields, only_fields)
        nfields = len(fields_to_plot)

        aspect = 1
        if not DataIs2DPolar(data):
            aspect = (data[x1].values.max() - data[x1].values.min()) / (
                data[x2].values.max() - data[x2].values.min()
            )
            aspect = aspect[()]
        else:
            aspect = 1.5

        ncols = max(1, int(math.floor(nfields * 1.5 * aspect / (1 + 1.5 * aspect))))
        nrows = max(1, int(math.ceil(nfields / ncols)))

        figsize0 = 3.0

        minmax = self._get_fields_minmax(data, fields_to_plot)

        kwargs = {}
        for fld in fields_to_plot:
            cmap = "viridis"
            if fld.startswith("N"):
                cmap = "inferno"
            elif fld.startswith("E"):
                cmap = "seismic"
            elif fld.startswith("B"):
                cmap = "BrBG"
            elif fld.startswith("J"):
                cmap = "coolwarm"
            if fld[0] in "EBJNT":
                mm = minmax[fld[0]]
                if mm is not None:
                    vmin, vmax = mm
                else:
                    raise ValueError(f"Field {fld} not found in minmax.")
            else:
                vmin, vmax = (
                    data[fld].min().values[()],
                    data[fld].max().values[()],
                )
                if vmin < 0:
                    if abs(vmin) > vmax:
                        vmax = abs(vmin)
                    else:
                        vmin = -vmax

            default_kwargs = {
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
            }
            kwargs[fld] = default_kwargs
            for fld_kwargs in plot_kwargs:
                if re.match(fld_kwargs, fld):
                    kwargs[fld] = {**default_kwargs, **plot_kwargs[fld_kwargs]}
                    break
            if "norm" in kwargs[fld]:
                vmin = kwargs[fld].pop("vmin")
                vmax = kwargs[fld].pop("vmax")
                norm_str: str = kwargs[fld].pop("norm")
                if norm_str == "linear":
                    kwargs[fld]["vmin"] = vmin
                    kwargs[fld]["vmax"] = vmax
                elif norm_str == "log":
                    if vmin <= 0:
                        vmin = 1e-3 * vmax
                    kwargs[fld]["norm"] = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                elif norm_str == "symlog":
                    linthresh = kwargs[fld].pop("linthresh", 1e-3 * vmax)
                    kwargs[fld]["norm"] = mcolors.SymLogNorm(
                        linthresh=linthresh, vmin=vmin, vmax=vmax, linscale=1
                    )

        def make_plot(ax: plt.Axes, fld: str):
            if DataIs2DPolar(data):
                data[fld].polar.pcolor(ax=ax, cbar_position=None, **kwargs[fld])
            else:
                data[fld].plot(ax=ax, add_colorbar=False, **kwargs[fld])

        def make_cbar(ax: plt.Axes, cbar: plt.Axes, fld: str):
            _ = cbar.set(xticks=[], xlabel=None, ylabel=None)
            cbar.yaxis.tick_right()
            vmin, vmax = ax.collections[0].get_clim()
            if vmin == vmax:
                vmin = -1
                vmax = 1
            data_norm = None
            coeff_pow = 0
            if abs(vmax) < 0.1 or abs(vmax) > 999:
                coeff_pow = int(np.log10(abs(vmax))) - 1
                coeff = 10**coeff_pow
                vmin /= coeff
                vmax /= coeff
            if isinstance(ax.collections[0].norm, mcolors.LogNorm):
                data_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                _ = cbar.set(ylim=(vmin, vmax), yscale="log")
                ys = np.logspace(np.log10(vmin), np.log10(vmax))
                _ = cbar.pcolor(
                    [0, 1],
                    ys,
                    np.transpose([ys] * 2),
                    cmap=kwargs[fld]["cmap"],
                    rasterized=True,
                    norm=data_norm,
                )
            elif isinstance(ax.collections[0].norm, mcolors.SymLogNorm):
                data_norm = ax.collections[0].norm
                _ = cbar.set_ylim(vmin, vmax)
                _ = cbar.set_yscale(
                    "symlog",
                    linthresh=data_norm.linthresh,
                    linscale=1,
                )
                ys = np.concatenate(
                    (
                        -np.logspace(
                            np.log10(-vmin),
                            np.log10(data_norm.linthresh),
                            num=100,
                            endpoint=False,
                        ),
                        np.linspace(
                            -data_norm.linthresh,
                            data_norm.linthresh,
                            num=10,
                            endpoint=False,
                        ),
                        np.logspace(
                            np.log10(data_norm.linthresh),
                            np.log10(vmax),
                            num=100,
                        ),
                    )
                )
                _ = cbar.pcolor(
                    [0, 1],
                    ys,
                    np.transpose([ys] * 2),
                    cmap=kwargs[fld]["cmap"],
                    rasterized=True,
                    norm=data_norm,
                )
            else:
                _ = cbar.set(ylim=(vmin, vmax))
                ys = np.linspace(vmin, vmax)
                _ = cbar.pcolor(
                    [0, 1],
                    ys,
                    np.transpose([ys] * 2),
                    cmap=kwargs[fld]["cmap"],
                    rasterized=True,
                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                )
            _ = ax.set(
                title=f"{fld}"
                + ("" if coeff_pow == 0 else f" [$\\cdot 10^{-coeff_pow}$]")
            )

        fig, axes = self._fixed_axes_grid_with_cbars(
            fields=fields_to_plot,
            makeplot=make_plot,
            makecbar=make_cbar,
            nrows=nrows,
            ncols=ncols,
            nfields=nfields,
            size=figsize0,
            aspect=aspect,
            pad=0.5,
            cbar_w=0.1,
            **fig_kwargs,
        )

        for n, ax in enumerate(axes):
            i = n // ncols
            j = n % ncols

            if j != 0:
                _ = ax.set(
                    ylabel=None,
                    yticklabels=[],
                )
            if (nfields - i * ncols - j) > ncols:
                _ = ax.set(
                    xlabel=None,
                    xticklabels=[],
                )
            if set_aspect is not None:
                _ = ax.set(aspect=set_aspect)

        _ = fig.suptitle(f"t = {data.t.values[()]:.2f}", y=1.0)
        return fig
