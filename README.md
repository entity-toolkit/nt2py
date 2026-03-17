# nt2.py

Python package for visualization and post-processing of the [`Entity`](https://github.com/entity-toolkit/entity) simulation data. For usage, please refer to the [documentation](https://entity-toolkit.github.io/wiki/content/2-howto/2-vis/#nt2py). The package is distributed via [`PyPI`](https://pypi.org/project/nt2py/):

```sh
pip install nt2py
```

## Usage

Simply pass the location to the data when initializing the main `Data` object:

```python
import nt2

data = nt2.Data("path/to/data")
```

The data is stored in specialized containers which can be accessed via corresponding attributes:

```python
data.fields      # < xr.Dataset
data.particles   # < special object which returns a pd.DataFrame when .load() is called
data.spectra     # < xr.Dataset
data.diagnostics # < pd.DataFrame
```

> If using Jupyter notebook, you can quickly preview the loaded metadata by simply running a cell with just `data` in it (or in regular python, by doing `print(data)`).

> Note, that by default, the `hdf5` support is disabled in `nt2py` (i.e., only `ADIOS2` format is supported). To enable it, install the package as `pip install "nt2py[hdf5]"` instead of simply `pip install nt2py`.

### Accessing the data

Fields and spectra are stored as lazily loaded `xarray` datasets (a collection of equal-sized arrays with shared axis coordinates). You may access the coordinates in each dimension using `.coords`:

```python
data.fields.coords
data.spectra.coords
```

Individual arrays can be requested by simply using, e.g., `data.fields.Ex` etc. One can also use slicing/selecting via the coordinates, i.e.,

```python 
data.fields.sel(t=5, method="nearest")
```

accesses all the fields at time `t=5` (using `method="nearest"` means it will take the closest time to value `5`). You may also access by index in each coordinate:

```python
data.fields.isel(x=-1)
```

accesses all the fields in the last position along the `x` coordinate. 

Note that all these operations do not load the actual data into memory; instead, the data is only loaded when explicitly requested (i.e., when plotting or explicitly calling `.values` or `.load()`.

Particles are stored in a special lazy container which acts very similar to `xarray`; you can still make selections using specific queries. For instance,

```python
data.particles.sel(sp=[1, 2, 4]).isel(t=-1)
```

selects all the particles of species 1, 2, and 4 on the last timestep. The loading of the data itself is done by calling: `.load()` method, which returns a simple `pandas` dataframe.

### Plotting

Plot a field (in Cartesian coordinates) at a specific time (or output step):

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
(data.fields.Ex * data.fields.Bx).sel(x=slice(None, 0.2)).movie.plot(name="ExBx")
```

For particles, one can also make 2D phase-space plots:

```python
data.particles.sel(sp=1).sel(t=1.0, method="nearest").phase_plot(
    x_quantity=lambda f: f.x,
    y_quantity=lambda f: f.ux,
    xy_bins=(np.linspace(0, 60, 100), np.linspace(-2, 2, 100)),
)
```

or a spectrum plot:

```python
data.particles.sel(sp=[1, 2]).sel(t=1.0, method="nearest").spectrum_plot()
```

You may also combine different quantities and plots (e.g., fields & particles) to produce a more customized movie:

```python
def plot(t, data):
    fig, ax = plt.subplots()
    data.fields.Ex.sel(t=t, method="nearest").sel(x=slice(None, 0.2)).plot(
        ax=ax, vmin=-0.001, vmax=0.001, cmap="BrBG"
    )
    prtls = data.particles.sel(t=t, method="nearest").load()
    ax.scatter(prtls.x, prtls.y, c="r" if prtls.sp == 1 else "b")
    ax.set_aspect(1)
data.makeMovie(plot)
```

You may also access the movie-making functionality directly in case you want to use it for other things:

```python
import nt2.plotters.export as nt2e

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

### Raw readers

In case you want to access the raw data without using `nt2py`'s `xarray`/`dask` lazy-loading, you may do so by using the readers. For example, for `ADIOS2` output data format:

```python
import nt2.readers.adios2 as nt2a

# define a reader
reader = nt2a.Reader()

# get all the valid steps for particles
valid_steps = reader.GetValidSteps("path/to/sim", "particles")

# get all variable names which have prefix "p" at the first valid step
variable_names = reader.ReadCategoryNamesAtTimestep(
    "path/to/sim", "particles", "p", valid_steps[0]
)

# convert the variable set into a list and take the first element
variable = list(variable_names)[0]

# read the actual array from the file
reader.ReadArrayAtTimestep(
    "path/to/sim", "particles", variable, valid_steps[0]
)
```

There are many more functions available within the reader. For `hdf5`, you can simply change the import to `nt2.readers.hdf5`, and the rest should remain the same. 


## CLI

Since version 1.0.0, `nt2py` also offers a command-line interface, accessed via `nt2` command. To view all the options, simply run:

```sh
nt2 --help
```

The plotting routine is pretty customizable. For instance, if the data is located in `myrun/mysimulation`, you can inspect the content of the data structure using:

```sh
nt2 show myrun/mysimulation
```

Or if you want to make a quick plot (a-la `inspect` discussed above) of the specific quantities, you may simply run:

```sh
nt2 plot myrun/mysimulation --fields "E.*;B.*" --isel "t=5" --sel "x=slice(-5, None); z=0.5"
```

This plots the 6-th snapshot (`t=5`) of all the `E` and `B` field components, sliced for `x > -5`, and at `z = 0.5` (notice, that you can use both `--isel` and `--sel`). If instead, you prefer to make a movie, simply do not specify the time:

```sh
nt2 plot myrun/mysimulation --fields "E.*;B.*" --sel "x=slice(-5, None); z=0.5"
```

> If you want to only install the CLI, without the library itself, you may do that via `pipx`: `pipx install nt2py`. 

## Features

1. Lazy loading and parallel processing of the simulation data with [`dask`](https://dask.org/).
2. Context-aware data manipulation with [`xarray`](http://xarray.pydata.org/en/stable/).
3. Parallel plotting and movie generation with [`loky`](https://pypi.org/project/loky/) and [`ffmpeg`](https://ffmpeg.org/).
4. Command-line interface, the `nt2` command, for quick plotting (both movies and snapshots).

## Testing

There are unit tests included with the code which also require downloading test data with [`git lfs`](https://git-lfs.com/) (installed separately from `git`). You may download the data simply by running `git lfs pull`.

## TODO

- [x] Unit tests
- [x] Plugins for other simulation data formats
- [ ] Support for sparse arrays for particles via `Sparse` library
- [x] Command-line interface
- [ ] Support for multiple runs
- [ ] Interactive regime (`hvplot`, `bokeh`, `panel`)
- [x] Ghost cells support
- [x] Usage examples
- [x] Parse the log file with timings
- [x] Raw reader
- [x] 3.14-compatible parallel output
