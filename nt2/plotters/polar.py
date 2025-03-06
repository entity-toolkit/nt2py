import numpy as np
from typing import Any

from nt2.containers.utils import _dataIs2DPolar


def DipoleSampling(**kwargs):
    """
    Returns an array of angles sampled from a dipole distribution.

    Parameters
    ----------
    nth : int, optional
        The number of angles to sample. Default is 30.
    pole : float, optional
        The fraction of the angles to sample from the poles. Default is 1/16.

    Returns
    -------
    ndarray
        An array of angles sampled from a dipole distribution.
    """
    nth = kwargs.get("nth", 30)
    pole = kwargs.get("pole", 1 / 16)

    nth_poles = int(nth * pole)
    nth_equator = (nth - 2 * nth_poles) // 2
    return np.concatenate(
        [
            np.linspace(0, np.pi * pole, nth_poles + 1)[1:],
            np.linspace(np.pi * pole, np.pi / 2, nth_equator + 2)[1:-1],
            np.linspace(np.pi * (1 - pole), np.pi, nth_poles + 1)[:-1],
        ]
    )


def MonopoleSampling(**kwargs):
    """
    Returns an array of angles sampled from a monopole distribution.

    Parameters
    ----------
    nth : int, optional
        The number of angles to sample. Default is 30.

    Returns
    -------
    ndarray
        An array of angles sampled from a monopole distribution.
    """
    nth = kwargs.get("nth", 30)

    return np.linspace(0, np.pi, nth + 2)[1:-1]


class _datasetPolarPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def pcolor(self, value, **kwargs):
        assert "t" not in self._obj[value].dims, "Time must be specified"
        assert _dataIs2DPolar(self._obj), "Data must be 2D polar"
        self._obj[value].polar.pcolor(**kwargs)

    def fieldplot(
        self,
        fr,
        fth,
        start_points=None,
        sample=None,
        invert_x=False,
        invert_y=False,
        **kwargs,
    ):
        """
        Plot field lines of a vector field defined by functions fr and fth.

        Parameters
        ----------
        fr : string
            Radial component of the vector field.
        fth : string
            Azimuthal component of the vector field.
        start_points : array_like, optional
            Starting points for the field lines. Either this or `sample` must be specified.
        sample : dict, optional
            Sampling template for generating starting points. Either this or `start_points` must be specified.
            The template can be "dipole" or "monopole". The dict also contains the starting `radius`,
            and the number of points in theta `nth` key.
        invert_x : bool, optional
            Whether to invert the x-axis. Default is False.
        invert_y : bool, optional
            Whether to invert the y-axis. Default is False.
        **kwargs :
            Additional keyword arguments passed to `fieldlines` and `ax.plot`.

        Raises
        ------
        ValueError
            If neither `start_points` nor `sample` are specified or if an unknown sampling template is given.

        Returns
        -------
        None

        Examples
        --------
        >>> ds.polar.fieldplot("Br", "Bth", sample={"template": "dipole", "nth": 30, "radius": 2.0})
        """
        import matplotlib.pyplot as plt

        if start_points is None and sample is None:
            raise ValueError("Either start_points or sample must be specified")
        elif start_points is None and sample is not None:
            radius = sample.pop("radius", 1.5)
            template = sample.pop("template", "dipole")
            if template == "dipole":
                start_points = [[radius, th] for th in DipoleSampling(**sample)]
            elif template == "monopole":
                start_points = [[radius, th] for th in MonopoleSampling(**sample)]
            else:
                raise ValueError("Unknown sampling template: " + template)

        fieldlines = self.fieldlines(fr, fth, start_points, **kwargs)
        ax = kwargs.pop("ax", plt.gca())
        for fieldline in fieldlines:
            if invert_x:
                fieldline[:, 0] = -fieldline[:, 0]
            if invert_y:
                fieldline[:, 1] = -fieldline[:, 1]
            ax.plot(*fieldline.T, **kwargs)

    def fieldlines(self, fr, fth, start_points, **kwargs):
        """
        Compute field lines of a vector field defined by functions fr and fth.

        Parameters
        ----------
        fr : string
            Radial component of the vector field.
        fth : string
            Azimuthal component of the vector field.
        start_points : array_like
            Starting points for the field lines.
        direction : str, optional
            Direction to integrate in. Can be "both", "forward" or "backward". Default is "both".
        stopWhen : callable, optional
            Function that takes the current position and returns True if the integration should stop. Default is to never stop.
        ds : float, optional
            Integration step size. Default is 0.1.
        maxsteps : int, optional
            Maximum number of integration steps. Default is 1000.

        Returns
        -------
        list
            List of field lines.

        Examples
        --------
        >>> ds.polar.fieldlines("Br", "Bth", [[2.0, np.pi / 4], [2.0, 3 * np.pi / 4]], stopWhen = lambda xy, rth: rth[0] > 5.0)
        """

        import numpy as np
        from scipy.interpolate import RegularGridInterpolator

        assert "t" not in self._obj[fr].dims, "Time must be specified"
        assert "t" not in self._obj[fth].dims, "Time must be specified"
        assert _dataIs2DPolar(self._obj), "Data must be 2D polar"

        useGreek = "θ" in self._obj.coords.keys()

        r, th = (
            self._obj.coords["r"].values,
            self._obj.coords["θ" if useGreek else "th"].values,
        )
        _, ths = np.meshgrid(r, th)
        fxs = self._obj[fr] * np.sin(ths) + self._obj[fth] * np.cos(ths)
        fys = self._obj[fr] * np.cos(ths) - self._obj[fth] * np.sin(ths)

        props: dict[str, Any] = {
            "method": "nearest",
            "bounds_error": False,
            "fill_value": 0,
        }
        interpFx = RegularGridInterpolator((th, r), fxs.values, **props)
        interpFy = RegularGridInterpolator((th, r), fys.values, **props)
        return [
            self._fieldline(interpFx, interpFy, rth, **kwargs) for rth in start_points
        ]

    def _fieldline(self, interp_fx, interp_fy, r_th_start, **kwargs):
        import numpy as np
        from copy import copy

        direction = kwargs.pop("direction", "both")
        stopWhen = kwargs.pop("stopWhen", lambda _, __: False)
        ds = kwargs.pop("ds", 0.1)
        maxsteps = kwargs.pop("maxsteps", 1000)

        rmax = self._obj.r.max()
        rmin = self._obj.r.min()

        def stop(xy, rth):
            return (
                stopWhen(xy, rth)
                or (rth[0] < rmin)
                or (rth[0] > rmax)
                or (rth[1] < 0)
                or (rth[1] > np.pi)
            )

        def integrate(delta, counter):
            r0, th0 = copy(r_th_start)
            XY = np.array([r0 * np.sin(th0), r0 * np.cos(th0)])
            RTH = [r0, th0]
            fieldline = np.array([XY])
            with np.errstate(divide="ignore", invalid="ignore"):
                while range(counter, maxsteps):
                    x, y = XY
                    r = np.sqrt(x**2 + y**2)
                    th = np.arctan2(-y, x) + np.pi / 2
                    RTH = [r, th]
                    vx = interp_fx((th, r))[()]
                    vy = interp_fy((th, r))[()]
                    vmag = np.sqrt(vx**2 + vy**2)
                    XY = XY + delta * np.array([vx, vy]) / vmag
                    if stop(XY, RTH) or np.isnan(XY).any() or np.isinf(XY).any():
                        break
                    else:
                        fieldline = np.append(fieldline, [XY], axis=0)
            return fieldline

        if direction == "forward":
            return integrate(ds, 0)
        elif direction == "backward":
            return integrate(-ds, 0)
        else:
            cntr = 0
            f1 = integrate(ds, cntr)
            f2 = integrate(-ds, cntr)
            return np.append(f2[::-1], f1, axis=0)


class _polarPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def pcolor(self, **kwargs):
        """
        Plots a pseudocolor plot of 2D polar data on a rectilinear projection.

        Parameters
        ----------
        ax : Axes object, optional
            The axes on which to plot. Default is the current axes.
        cell_centered : bool, optional
            Whether the data is cell-centered. Default is True.
        cell_size : float, optional
            If not cell_centered, defines the fraction of the cell to use for coloring. Default is 0.75.
        cbar_size : str, optional
            The size of the colorbar. Default is "5%".
        cbar_pad : float, optional
            The padding between the colorbar and the plot. Default is 0.05.
        cbar_position : str, optional
            The position of the colorbar. Default is "right".
        cbar_ticksize : int or float, optional
            The size of the ticks on the colorbar. Default is None.
        title : str, optional
            The title of the plot. Default is None.
        invert_x : bool, optional
            Whether to invert the x-axis. Default is False.
        invert_y : bool, optional
            Whether to invert the y-axis. Default is False.
        ylabel : str, optional
            The label for the y-axis. Default is "y".
        xlabel : str, optional
            The label for the x-axis. Default is "x".
        label : str, optional
            The label for the plot. Default is None.

        Returns
        -------
        matplotlib.collections.Collection
            The pseudocolor plot.

        Raises
        ------
        AssertionError
            If `ax` is a polar projection or if time is not specified or if data is not 2D polar.

        Notes
        -----
        Additional keyword arguments are passed to `pcolormesh`.
        """

        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib import tri
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        useGreek = "θ" in self._obj.coords.keys()

        ax = kwargs.pop("ax", plt.gca())
        cbar_size = kwargs.pop("cbar_size", "5%")
        cbar_pad = kwargs.pop("cbar_pad", 0.05)
        cbar_pos = kwargs.pop("cbar_position", "right")
        cbar_orientation = (
            "vertical" if cbar_pos == "right" or cbar_pos == "left" else "horizontal"
        )
        cbar_ticksize = kwargs.pop("cbar_ticksize", None)
        title = kwargs.pop("title", None)
        invert_x = kwargs.pop("invert_x", False)
        invert_y = kwargs.pop("invert_y", False)
        ylabel = kwargs.pop("ylabel", "y")
        xlabel = kwargs.pop("xlabel", "x")
        label = kwargs.pop("label", None)
        cell_centered = kwargs.pop("cell_centered", True)
        cell_size = kwargs.pop("cell_size", 0.75)

        assert ax.name != "polar", "`ax` must be a rectilinear projection"
        assert "t" not in self._obj.dims, "Time must be specified"
        assert _dataIs2DPolar(self._obj), "Data must be 2D polar"
        ax.grid(False)
        if type(kwargs.get("norm", None)) is colors.LogNorm:
            cm = kwargs.get("cmap", "viridis")
            cm = mpl.colormaps[cm]
            cm.set_bad(cm(0))
            kwargs["cmap"] = cm

        vals = self._obj.values.flatten()
        vals = np.concatenate((vals, vals))
        if not cell_centered:
            drs = self._obj.coords["r_2"] - self._obj.coords["r_1"]
            dths = (
                self._obj.coords["θ_2" if useGreek else "th_2"]
                - self._obj.coords["θ_1" if useGreek else "th_1"]
            )
            r1s = self._obj.coords["r_1"] - drs * cell_size / 2
            r2s = self._obj.coords["r_1"] + drs * cell_size / 2
            th1s = (
                self._obj.coords["θ_1" if useGreek else "th_1"] - dths * cell_size / 2
            )
            th2s = (
                self._obj.coords["θ_1" if useGreek else "th_1"] + dths * cell_size / 2
            )
            rs = np.ravel(np.column_stack((r1s, r2s)))
            ths = np.ravel(np.column_stack((th1s, th2s)))
            nr = len(rs)
            nth = len(ths)
            rs, ths = np.meshgrid(rs, ths)
            rs = rs.flatten()
            ths = ths.flatten()
            points_1 = np.arange(nth * nr).reshape(nth, -1)[:-1:2, :-1:2].flatten()
            points_2 = np.arange(nth * nr).reshape(nth, -1)[:-1:2, 1::2].flatten()
            points_3 = np.arange(nth * nr).reshape(nth, -1)[1::2, 1::2].flatten()
            points_4 = np.arange(nth * nr).reshape(nth, -1)[1::2, :-1:2].flatten()

        else:
            rs = np.append(self._obj.coords["r_1"], self._obj.coords["r_2"][-1])
            ths = np.append(
                self._obj.coords["θ_1" if useGreek else "th_1"],
                self._obj.coords["θ_2" if useGreek else "th_2"][-1],
            )
            nr = len(rs)
            nth = len(ths)
            rs, ths = np.meshgrid(rs, ths)
            rs = rs.flatten()
            ths = ths.flatten()
            points_1 = np.arange(nth * nr).reshape(nth, -1)[:-1, :-1].flatten()
            points_2 = np.arange(nth * nr).reshape(nth, -1)[:-1, 1:].flatten()
            points_3 = np.arange(nth * nr).reshape(nth, -1)[1:, 1:].flatten()
            points_4 = np.arange(nth * nr).reshape(nth, -1)[1:, :-1].flatten()
        x, y = rs * np.sin(ths), rs * np.cos(ths)
        if invert_x:
            x = -x
        if invert_y:
            y = -y
        triang = tri.Triangulation(
            x,
            y,
            triangles=np.concatenate(
                [
                    np.array([points_1, points_2, points_3]).T,
                    np.array([points_1, points_3, points_4]).T,
                ],
                axis=0,
            ),
        )
        ax.set(
            aspect="equal",
            xlabel=xlabel,
            ylabel=ylabel,
        )
        im = ax.tripcolor(triang, vals, rasterized=True, shading="flat", **kwargs)
        if cbar_pos is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(cbar_pos, size=cbar_size, pad=cbar_pad)
            _ = plt.colorbar(
                im,
                cax=cax,
                label=self._obj.name if label is None else label,
                orientation=cbar_orientation,
            )
            if cbar_orientation == "vertical":
                axis = cax.yaxis
            else:
                axis = cax.xaxis
            axis.set_label_position(cbar_pos)
            axis.set_ticks_position(cbar_pos)
            if cbar_ticksize is not None:
                cax.tick_params("both", labelsize=cbar_ticksize)
        ax.set_title(
            f"t={self._obj.coords['t'].values[()]:.2f}" if title is None else title
        )
        return im

    def contour(self, **kwargs):
        """
        Plots a pseudocolor plot of 2D polar data on a rectilinear projection.

        Parameters
        ----------
        ax : Axes object, optional
            The axes on which to plot. Default is the current axes.
        invert_x : bool, optional
            Whether to invert the x-axis. Default is False.
        invert_y : bool, optional
            Whether to invert the y-axis. Default is False.

        Returns
        -------
        matplotlib.contour.QuadContourSet
            The contour plot.

        Raises
        ------
        AssertionError
            If `ax` is a polar projection or if time is not specified or if data is not 2D polar.

        Notes
        -----
        Additional keyword arguments are passed to `contour`.
        """

        import warnings
        import matplotlib.pyplot as plt

        useGreek = "θ" in self._obj.coords.keys()

        ax = kwargs.pop("ax", plt.gca())
        title = kwargs.pop("title", None)
        invert_x = kwargs.pop("invert_x", False)
        invert_y = kwargs.pop("invert_y", False)

        assert ax.name != "polar", "`ax` must be a rectilinear projection"
        assert "t" not in self._obj.dims, "Time must be specified"
        assert _dataIs2DPolar(self._obj), "Data must be 2D polar"
        ax.grid(False)
        r, th = np.meshgrid(
            self._obj.coords["r"], self._obj.coords["θ" if useGreek else "th"]
        )
        x, y = r * np.sin(th), r * np.cos(th)
        if invert_x:
            x = -x
        if invert_y:
            y = -y
        ax.set(
            aspect="equal",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.contour(x, y, self._obj.values, **kwargs)

        ax.set_title(
            f"t={self._obj.coords['t'].values[()]:.2f}" if title is None else title
        )
        return im
