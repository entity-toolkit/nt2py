from nt2.export import _makeFramesAndMovie


class _moviePlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(self, name, movie_kwargs={}, *args, **kwargs):
        if "t" not in self._obj.dims:
            raise ValueError("The dataset does not have a time dimension.")

        import matplotlib.pyplot as plt

        def plot_func(ti, _):
            if len(self._obj.isel(t=ti).dims) == 2:
                x1, x2 = self._obj.isel(t=ti).dims
                nx1, nx2 = len(self._obj.isel(t=ti)[x1]), len(self._obj.isel(t=ti)[x2])
                aspect = nx1 / nx2
                plt.figure(figsize=(6, 4 * aspect))
            self._obj.isel(t=ti).plot(*args, **kwargs)
            if len(self._obj.isel(t=ti).dims) == 2:
                plt.gca().set_aspect("equal")
            plt.tight_layout()

        num_cpus = movie_kwargs.pop("num_cpus", None)
        return _makeFramesAndMovie(
            name=name,
            data=self._obj,
            plot=plot_func,
            times=list(range(len(self._obj.t))),
            num_cpus=num_cpus,
            **movie_kwargs,
        )
