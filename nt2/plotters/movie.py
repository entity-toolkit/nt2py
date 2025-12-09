from typing import Any
from nt2.plotters.export import (
    makeFramesAndMovie,
)
import xarray as xr


class accessor:
    """
    Movie plotter for xarray DataArray objects.

    Functions
    ---------
    plot(name: str, movie_kwargs: dict[str, Any] | None = None, *args: Any, **kwargs: Any) -> bool
        Plot a movie of the DataArray over the time dimension 't'.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj: xr.DataArray = xarray_obj

    def plot(
        self,
        name: str,
        movie_kwargs: dict[str, Any] = {},
        fig_kwargs: dict[str, Any] = {},
        aspect_equal: bool = False,
        **kwargs: Any,
    ) -> bool:
        """
        Plot a movie of the DataArray over the time dimension 't'.

        Parameters
        ----------
        name : str
            The name of the output movie file.
        movie_kwargs : dict[str, Any], optional
            Additional keyword arguments for movie creation (default is {}).
        fig_kwargs : dict[str, Any], optional
            Additional keyword arguments for figure creation (default is {}).
        aspect_equal : bool, optional
            Whether to set equal aspect ratio for 2D plots (default is False).
        **kwargs : Any
            Additional keyword arguments to pass to the plotting function.
        Returns
        -------
        bool
            True if the movie was created successfully, False otherwise.

        Notes
        -----
        kwargs are passed to the xarray plotting function:
        ```
        xarray.plot(**kwargs)
        ```
        """
        if "t" not in self._obj.dims:
            raise ValueError("The dataset does not have a time dimension.")

        import matplotlib.pyplot as plt

        def plot_func(ti: int, _: Any) -> None:
            if len(self._obj.isel(t=ti).dims) == 2:
                if aspect_equal:
                    x1, x2 = self._obj.isel(t=ti).dims
                    nx1, nx2 = len(self._obj.isel(t=ti)[x1]), len(
                        self._obj.isel(t=ti)[x2]
                    )
                    aspect = nx1 / nx2
                    figsize = fig_kwargs.get("figsize", (6, 4 * aspect))
                    _ = plt.figure(figsize=figsize, **fig_kwargs)
                else:
                    _ = plt.figure(**fig_kwargs)
            self._obj.isel(t=ti).plot(**kwargs)
            if aspect_equal and len(self._obj.isel(t=ti).dims) == 2:
                plt.gca().set_aspect("equal")
            plt.tight_layout()

        num_cpus: int | None = movie_kwargs.pop("num_cpus", None)
        return makeFramesAndMovie(
            name=name,
            data=self._obj,
            plot=plot_func,
            times=list(range(len(self._obj.t))),
            num_cpus=num_cpus,
            **movie_kwargs,
        )
