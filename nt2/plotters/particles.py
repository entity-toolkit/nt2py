import xarray as xr
import numpy as np


class ds_accessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj: xr.Dataset = xarray_obj

    def phaseplot(
        self,
        x: str = "x",
        y: str = "ux",
        xbins: None | np.ndarray = None,
        ybins: None | np.ndarray = None,
        xlims: None | tuple[float] = None,
        ylims: None | tuple[float] = None,
        xnbins: int = 100,
        ynbins: int = 100,
        **kwargs,
    ):
        """
        Create a 2D histogram (phase plot) of two variables in the dataset.

        Parameters
        ----------
        x : str
            The variable name for the x-axis (default: "x").
        y : str
            The variable name for the y-axis (default: "ux").
        xbins : np.ndarray, optional
            The bin edges for the x-axis. If None, 100 bins between min and max of x are used.
        ybins : np.ndarray, optional
            The bin edges for the y-axis. If None, 100 bins between min and max of y are used.
        xlims : tuple[float], optional
            The limits for the x-axis. If None, the limits are determined from the data.
        ylims : tuple[float], optional
            The limits for the y-axis. If None, the limits are determined from the data.
        xnbins : int, optional
            The number of bins for the x-axis if xbins is None (default: 100).
        ynbins : int, optional
            The number of bins for the y-axis if ybins is None (default: 100).
        **kwargs
            Additional keyword arguments passed to matplotlib's pcolormesh.

        Raises
        ------
        AssertionError
            If x or y are not valid variable names in the dataset, or if the dataset has a time dimension.

        Returns
        -------
        None

        Examples
        --------
        >>> ds.phaseplot(x='x', y='ux', xbins=np.linspace(0, 1000, 100), ybins=np.linspace(-5, 5, 50))
        """
        assert x in list(self._obj.keys()) and y in list(
            self._obj.keys()
        ), "x and y must be valid variable names in the dataset"
        assert (
            len(self._obj[x].dims) == 1 and len(self._obj[y].dims) == 1
        ), "x and y must be 1D variables"
        assert "t" not in self._obj.dims, "Dataset must not have time dimension"

        import matplotlib.pyplot as plt

        if xbins is None:
            if xlims is not None:
                xbins_ = np.linspace(xlims[0], xlims[1], xnbins)
            else:
                xbins_ = np.linspace(
                    self._obj[x].values.min(), self._obj[x].values.max(), xnbins
                )
        else:
            xbins_ = xbins
        if ybins is None:
            if ylims is not None:
                ybins_ = np.linspace(ylims[0], ylims[1], ynbins)
            else:
                ybins_ = np.linspace(
                    self._obj[y].values.min(), self._obj[y].values.max(), ynbins
                )
        else:
            ybins_ = ybins

        cnt, _, _ = np.histogram2d(
            self._obj[x].values, self._obj[y].values, bins=[xbins_, ybins_]
        )
        xbins_ = 0.5 * (xbins_[1:] + xbins_[:-1])
        ybins_ = 0.5 * (ybins_[1:] + ybins_[:-1])

        ax = kwargs.pop("ax", plt.gca())
        ax.pcolormesh(
            xbins_,
            ybins_,
            cnt.T,
            rasterized=True,
            **kwargs,
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
