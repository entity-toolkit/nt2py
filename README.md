## nt2

Python package for visualization and post-processing of the [`Entity`](https://github.com/entity-toolkit/entity) simulation data. For usage, please refer to the [documentation](https://entity-toolkit.github.io/entity/howto/vis/#nt2py).

### Features

1. Lazy loading and parallel processing of the simulation data with [`dask`](https://dask.org/).
2. Context-aware data manipulation with [`xarray`](http://xarray.pydata.org/en/stable/).
3. Parellel plotting and movie generation with [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) and [`ffmpeg`](https://ffmpeg.org/).

### TODO

- [ ] Unit tests
- [ ] Plugins for other simulation data formats
- [ ] Usage examples