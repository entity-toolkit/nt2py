from nt2.containers.utils import _dataIs2DPolar
from nt2.export import _makeFramesAndMovie


class _datasetInspectPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(
        self,
        fig=None,
        name=None,
        skip_fields=[],
        only_fields=[],
        fig_kwargs={},
        plot_kwargs={},
        movie_kwargs={},
    ):
        """
        Plots the overview plot for fields at a given time or step (or as a movie).

        Kwargs
        ------
        fig : matplotlib.figure.Figure, optional
            The figure to plot the data (if None, a new figure is created). Default is None.

        name : string, optional
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

        Returns
        -------
        figure : matplotlib.figure.Figure | boolean
            The figure with the plotted data (if single timestep) or True/False.

        """
        if "t" in self._obj.dims:
            if name is None:
                raise ValueError(
                    "Please provide a name for saving the frames and movie"
                )

            def plot_func(ti, _):
                self.plot_frame(
                    self._obj.isel(t=ti),
                    None,
                    skip_fields,
                    only_fields,
                    fig_kwargs,
                    plot_kwargs,
                )

            return _makeFramesAndMovie(
                name=name,
                data=self._obj,
                plot=plot_func,
                times=list(range(len(self._obj.t))),
                **movie_kwargs,
            )
        else:
            return self.plot_frame(
                self._obj, fig, skip_fields, only_fields, fig_kwargs, plot_kwargs
            )

    def plot_frame(self, data, fig, skip_fields, only_fields, fig_kwargs, plot_kwargs):
        if len(data.dims) != 2:
            raise ValueError("Pass 2D data; use .sel or .isel to reduce dimension.")

        x1, x2 = data.dims

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        import matplotlib.colors as mcolors
        import numpy as np
        import re
        import math

        # count the number of subplots
        nfields = len(data.data_vars)
        if nfields > 0:
            if len(only_fields) == 0:
                fields_to_plot = [
                    f
                    for f in list(data.keys())
                    if not any([re.match(sf, f) for sf in skip_fields])
                ]
            else:
                fields_to_plot = [
                    f
                    for f in list(data.keys())
                    if any([re.match(sf, f) for sf in only_fields])
                ]
        else:
            fields_to_plot = []

        if fields_to_plot == []:
            raise ValueError("No fields to plot.")

        nfields = len(fields_to_plot)

        aspect = 1
        if _dataIs2DPolar(data):
            aspect = 0.5
        else:
            aspect = len(data[x1]) / len(data[x2])

        ncols = 3 if aspect <= 1.15 else int(math.ceil(nfields / 3))
        nrows = 3 if aspect > 1.15 else int(math.ceil(nfields / 3))

        figsize0 = 3

        if fig is None:
            dpi = fig_kwargs.pop("dpi", 200)
            fig = plt.figure(
                figsize=(
                    figsize0 * ncols * aspect * (1 + 0.2 / aspect),
                    figsize0 * nrows,
                ),
                dpi=dpi,
                **fig_kwargs,
            )

        gs = GridSpec(nrows, ncols, wspace=0.2 / aspect)
        gs_for_axes = [
            GridSpecFromSubplotSpec(
                1,
                2,
                subplot_spec=gs[i],
                width_ratios=[1, max(0.025 / aspect, 0.025)],
                wspace=0.01,
            )
            for i in range(nfields)
        ]
        if aspect <= 1.15:
            axes = [
                fig.add_subplot(gs_for_axes[i * ncols + j][0])
                for i in range(nrows)
                for j in range(ncols)
                if i * ncols + j < nfields
            ]
            cbars = [
                fig.add_subplot(gs_for_axes[i * ncols + j][1])
                for i in range(nrows)
                for j in range(ncols)
                if i * ncols + j < nfields
            ]
        else:
            axes = [
                fig.add_subplot(gs_for_axes[i * ncols + j][0])
                for j in range(ncols)
                for i in range(nrows)
                if i * ncols + j < nfields
            ]
            cbars = [
                fig.add_subplot(gs_for_axes[i * ncols + j][1])
                for j in range(ncols)
                for i in range(nrows)
                if i * ncols + j < nfields
            ]

        # find minmax for all components
        minmax: dict[str, None | tuple] = {
            "E": None,
            "B": None,
            "J": None,
            "N": None,
            "T": None,
        }
        for fld in fields_to_plot:
            vmin, vmax = (
                data[fld].min().values[()],
                data[fld].max().values[()],
            )
            if fld[0] in "EBJNT":
                if minmax[fld[0]] is None:
                    minmax[fld[0]] = (vmin, vmax)
                else:
                    minmax[fld[0]] = (
                        min(minmax[fld[0]][0], vmin),
                        max(minmax[fld[0]][1], vmax),
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
                if minmax[fld[0]] is not None:
                    vmin, vmax = minmax[fld[0]]
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
                kwargs[fld].pop("vmin")
                kwargs[fld].pop("vmax")

        if _dataIs2DPolar(data):
            raise NotImplementedError("Polar plots for inspect not implemented yet.")
        else:
            for fld, ax in zip(fields_to_plot, axes):
                data[fld].plot(ax=ax, add_colorbar=False, **kwargs[fld])

        for i, (ax, cbar, fld) in enumerate(zip(axes, cbars, fields_to_plot)):
            cbar.set(xticks=[], xlabel=None, ylabel=None)
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
                cbar.set(ylim=(vmin, vmax), yscale="log")
                data_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                ys = np.logspace(np.log10(vmin), np.log10(vmax))
                cbar.pcolor(
                    [0, 1],
                    ys,
                    np.transpose([ys] * 2),
                    cmap=kwargs[fld]["cmap"],
                    rasterized=True,
                    norm=data_norm,
                )
            elif isinstance(ax.collections[0].norm, mcolors.SymLogNorm):
                raise NotImplementedError("SymLogNorm not implemented yet.")
            else:
                cbar.set(ylim=(vmin, vmax))
                ys = np.linspace(vmin, vmax)
                cbar.pcolor(
                    [0, 1],
                    ys,
                    np.transpose([ys] * 2),
                    cmap=kwargs[fld]["cmap"],
                    rasterized=True,
                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                )
            ax.set(
                title=f"{fld}{'' if coeff_pow == 0 else fr' [$\cdot 10^{-coeff_pow}$]'}"
            )

        for n, ax in enumerate(axes):
            if aspect > 1.15:
                i = n % nrows
                j = n // nrows
            else:
                i = n // ncols
                j = n % ncols

            if j != 0:
                ax.set(
                    ylabel=None,
                    yticklabels=[],
                )
            if (nfields - i * ncols - j) > ncols:
                ax.set(
                    xlabel=None,
                    xticklabels=[],
                )
            ax.set(aspect=1)

        fig.suptitle(f"t = {data.t.values[()]:.2f}", y=0.95)
        return fig
