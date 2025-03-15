## nt2.py

Python package for visualization and post-processing of the [`Entity`](https://github.com/entity-toolkit/entity) simulation data. For usage, please refer to the [documentation](https://entity-toolkit.github.io/wiki/getting-started/vis/#nt2py). The package is distributed via [`PyPI`](https://pypi.org/project/nt2py/):

```sh
pip install nt2py
```

### Usage

The Library works both with single-file output as well as with separate files. In either case, the location of the data is passed via `path` keyword argument.

```python
import nt2

data = nt2.Data(path="path/to/data")
# example: 
#   data = nt2.Data(path="path/to/shock.h5") : for single-file
#   data = nt2.Data(path="path/to/shock") : for multi-file
```

The data is stored in specialized containers which can be accessed via corresponding attributes:

```python
data.fields     # < xr.Dataset
data.particles  # < dict[int : xr.Dataset]
data.spectra    # < xr.Dataset
```

> If using Jupyter notebook, you can quickly preview the loaded metadata by simply running a cell with just `data` in it (or in regular python, by doing `print(data)`).

#### Examples

Plot a field (in cartesian space) at a specific time (or output step):

```python
data.fields.Ex.sel(t=10.0, method="nearest").plot() # time ~ 10
data.fields.Ex.isel(t=5).plot()                     # output step = 5
```

Plot a slice or time-averaged field quantities:

```python
data.fields.Bz.mean("t").plot()
data.fields.Bz.sel(t=10.0, x=0.5, method="nearest").plot()
```

Plot in spherical coordinates (+ combine several fields):

```python
e_dot_b = (data.fields.Er * data.fields.Br +\
           data.fields.Eth * data.fields.Bth +\
           data.fields.Eph * data.fields.Bph)
bsqr = data.fields.Br**2 + data.fields.Bth**2 + data.fields.Bph**2
# only plot radial extent of up to 10
(e_dot_b / bsqr).sel(t=50.0, method="nearest").sel(r=slice(None, 10)).polar.pcolor()
```

You can also quickly plot the fields at a specific time using the handy `.inspect` accessor:

```python
data.fields\
    .sel(t=3.0, method="nearest")\
    .sel(x=slice(-0.2, 0.2))\
    .inspect.plot(only_fields=["E", "B"])
# Hint: use `<...>.plot?` to see all options
```

Or if no time is specified, it will create a quick movie (need to also provide a `name` in that case):

```python
data.fields\
    .sel(x=slice(-0.2, 0.2))\
    .inspect.plot(name="inspect", only_fields=["E", "B", "N"])
```

You can also create a movie of a single field quantity (can be custom):

```python
(data.fields.Ex * data.fields.Bx).sel(x=slice(None, 0.2)).movie.plot(name="ExBx", vmin=-0.01, vmax=0.01, cmap="BrBG")
```

You may also combine different quantities and plots (e.g., fields & particles) to produce a more customized movie:

```python
def plot(t, data):
    fig, ax = mpl.pyplot.subplots()
    data.fields.Ex.sel(t=t, method="nearest").sel(x=slice(None, 0.2)).plot(
        ax=ax, vmin=-0.001, vmax=0.001, cmap="BrBG"
    )
    for sp in range(1, 3):
        ax.scatter(
            data.particles[sp].sel(t=t, method="nearest").x,
            data.particles[sp].sel(t=t, method="nearest").y,
            c="r" if sp == 1 else "b",
        )
    ax.set_aspect(1)
data.makeMovie(plot)
```

You may also access the movie-making functionality directly in case you want to use it for other things:

```python
import nt2.export as nt2e

def plot(t):
  ...

#             this will be the array of `t`-s passed to `plot`
#                           |
#                           V
nt2e.makeFrames(plot, np.arange(100), "myAnim")
nt2e.makeMovie(
    input="myAnim/", output="myAnim.mp4", number=5, overwrite=True
)

# or combined together
nt2e.makeFramesAndMovie(
    name="myAnim", plot=plot, times=np.arange(100)
)
```

#### Plots for debugging

If the simulation also outputs the ghost cells, `nt2py` will interpret the fields differently, and instead of reading the physical coordinates, will build the coordinates based on the number of cells (including ghost cells). In particular, instead of, e.g., `data.fields.x` it will contain `data.fields.i1`. The data will also contain information about the meshblock decomposition. For instance, if you have `Nx` meshblocks in the `x` direction, each having `nx` cells, the coordinates `data.fields.i1` will go from `0` to `nx * NX + 2 * NGHOSTS * Nx`.

You can overplot both the coordinate grid as well as the active zones of the meshblocks using the following:

```python
ax = plt.gca()
data.fields.Ex.isel(t=ti).plot(ax=ax)
data.plotGrid(ax=ax)
data.plotDomains(ax=ax)
```

> Keep in mind, that by default `Entity` converts all quantities to tetrad basis (or contravariant in GR) and interpolates to cell centers before outputting (except for the ghost cells). So when doing plots for debugging, make sure to also set `as_is = true` (together with `ghosts = true`) in the `[output.debug]` section of the `toml` input file. This will ensure the fields are being output as is, with no conversion or interpolation. This does not apply to particle moments, which are never stored in the code and are computed only during the output.

### Dashboard

Support for the dask dashboard is still in beta, but you can access it by first launching the dashboard client:

```python
import nt2 
nt2.Dashboard()
```

This will output the port where the dashboard server is running, e.g., `Dashboard: http://127.0.0.1:8787/status`. Click on it (or enter in your browser) to open the dashboard.

### Features

1. Lazy loading and parallel processing of the simulation data with [`dask`](https://dask.org/).
2. Context-aware data manipulation with [`xarray`](http://xarray.pydata.org/en/stable/).
3. Parellel plotting and movie generation with [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) and [`ffmpeg`](https://ffmpeg.org/).

### TODO

- [ ] Unit tests
- [ ] Plugins for other simulation data formats
- [ ] Support for multiple runs
- [ ] Interactive regime (`hvplot`, `bokeh`, `panel`)
- [x] Ghost cells support
- [x] Usage examples
