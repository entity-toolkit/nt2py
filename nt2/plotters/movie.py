from typing import Any
from nt2.plotters.export import (
    makeFramesAndMovie,
)
import xarray as xr


class accessor:
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj: xr.DataArray = xarray_obj

    def plot(
        self,
        name: str,
        movie_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        if movie_kwargs is None:
            movie_kwargs = {}
        if "t" not in self._obj.dims:
            raise ValueError("The dataset does not have a time dimension.")

        import matplotlib.pyplot as plt

        def plot_func(ti: int, _: Any) -> None:
            if len(self._obj.isel(t=ti).dims) == 2:
                x1, x2 = self._obj.isel(t=ti).dims
                nx1, nx2 = len(self._obj.isel(t=ti)[x1]), len(self._obj.isel(t=ti)[x2])
                aspect = nx1 / nx2
                _ = plt.figure(figsize=(6, 4 * aspect))
            self._obj.isel(t=ti).plot(*args, **kwargs)
            if len(self._obj.isel(t=ti).dims) == 2:
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
