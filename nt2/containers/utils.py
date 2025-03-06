import h5py
import numpy as np
import xarray as xr
from dask.array.core import from_array
from dask.array.core import stack
import inspect


def InheritClassDocstring(cls):
    if cls.__doc__ is None:
        cls.__doc__ = ""
    for base in inspect.getmro(cls):
        if base.__doc__ is not None:
            cls.__doc__ += base.__doc__
    return cls


def _dataIs2DPolar(ds):
    return ("r" in ds.dims and ("θ" in ds.dims or "th" in ds.dims)) and len(
        ds.dims
    ) == 2


def _read_category_metadata(
    single_file: bool, prefix: str, file: h5py.File | list[h5py.File]
):
    outsteps = []
    steps = []
    times = []
    quantities = None
    for i, st in enumerate(file):
        if single_file:
            assert isinstance(file, h5py.File)
            group = file[st]
        else:
            assert isinstance(file[i], h5py.File)
            group = st["Step0"]
        if isinstance(group, h5py.Group):
            if any([k.startswith(prefix) for k in group if k is not None]):
                if quantities is None:
                    quantities = [k for k in group.keys() if k.startswith(prefix)]
                outsteps.append(st if single_file else f"Step{i}")
                time_ds = group["Time"]
                if isinstance(time_ds, h5py.Dataset):
                    times.append(time_ds[()])
                else:
                    raise ValueError(f"Unexpected type {type(time_ds)}")
                step_ds = group["Step"]
                if isinstance(step_ds, h5py.Dataset):
                    steps.append(int(step_ds[()]))
                else:
                    raise ValueError(f"Unexpected type {type(step_ds)}")
        else:
            raise ValueError("Unexpected type")
    outsteps = sorted(outsteps, key=lambda x: int(x.replace("Step", "")))
    steps = sorted(steps)
    times = np.array(sorted(times), dtype=np.float64)
    return {
        "quantities": quantities,
        "outsteps": outsteps,
        "steps": steps,
        "times": times,
    }


# fields
def _read_coordinates(coords: list[str], file: h5py.File):
    for st in file:
        group = file[st]
        if isinstance(group, h5py.Group):
            if any([k.startswith("X") for k in group if k is not None]):
                # cell-centered coords
                xc = {
                    c: (
                        np.asarray(xi[:])
                        if isinstance(xi := group[f"X{i+1}"], h5py.Dataset) and xi
                        else None
                    )
                    for i, c in enumerate(coords[::-1])
                }
                # cell edges
                xe_min = {
                    f"{c}_1": (
                        c,
                        (
                            np.asarray(xi[:-1])
                            if isinstance((xi := group[f"X{i+1}e"]), h5py.Dataset)
                            else None
                        ),
                    )
                    for i, c in enumerate(coords[::-1])
                }
                xe_max = {
                    f"{c}_2": (
                        c,
                        (
                            np.asarray(xi[1:])
                            if isinstance((xi := group[f"X{i+1}e"]), h5py.Dataset)
                            else None
                        ),
                    )
                    for i, c in enumerate(coords[::-1])
                }
                return {"x_c": xc, "x_emin": xe_min, "x_emax": xe_max}
        else:
            raise ValueError(f"Unexpected type {type(file[st])}")
    raise ValueError("Could not find coordinates in file")


def _preload_field(
    single_file: bool,
    k: str,
    dim: int,
    ngh: int,
    outsteps: list[int],
    times: list[float],
    steps: list[int],
    coords: list[str],
    xc_coords: dict[str, str],
    xe_min_coords: dict[str, str],
    xe_max_coords: dict[str, str],
    coord_replacements: list[tuple[str, str]],
    field_replacements: list[tuple[str, str]],
    layout: str,
    file: h5py.File | list[h5py.File],
):
    if dim == 1:
        noghosts = slice(ngh, -ngh) if ngh > 0 else slice(None)
    elif dim == 2:
        noghosts = (slice(ngh, -ngh), slice(ngh, -ngh)) if ngh > 0 else slice(None)
    elif dim == 3:
        noghosts = (
            (slice(ngh, -ngh), slice(ngh, -ngh), slice(ngh, -ngh))
            if ngh > 0
            else slice(None)
        )
    else:
        raise ValueError("Invalid dimension")

    dask_arrays = []
    if single_file:
        for s in outsteps:
            assert isinstance(file, h5py.File)
            dset = file[f"{s}/{k}"]
            if isinstance(dset, h5py.Dataset):
                array = from_array(np.transpose(dset) if layout == "right" else dset)
                dask_arrays.append(array[noghosts])
            else:
                raise ValueError(f"Unexpected type {type(dset)}")
    else:
        for f in file:
            assert isinstance(f, h5py.File)
            dset = f[f"Step0/{k}"]
            if isinstance(dset, h5py.Dataset):
                array = from_array(np.transpose(dset) if layout == "right" else dset)
                dask_arrays.append(array[noghosts])
            else:
                raise ValueError(f"Unexpected type {type(dset)}")

    k_ = k[1:]
    for c in coord_replacements:
        if "_" not in k_:
            k_ = k_.replace(c[0], c[1])
        else:
            k_ = "_".join([k_.split("_")[0].replace(c[0], c[1])] + k_.split("_")[1:])
    for f in field_replacements:
        k_ = k_.replace(*f)

    return k_, xr.DataArray(
        stack(dask_arrays, axis=0),
        dims=["t", *coords],
        name=k_,
        coords={
            "t": times,
            "s": ("t", steps),
            **xc_coords,
            **xe_min_coords,
            **xe_max_coords,
        },
    )


# particles
def _list_to_ragged(arr):
    max_len = np.max([len(a) for a in arr])
    return map(
        lambda a: np.concatenate([a, np.full(max_len - len(a), np.nan)]),
        arr,
    )


def _read_particle_species(first_step: str, file: h5py.File):
    group = file[first_step]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Unexpected type {type(group)}")
    species = np.unique(
        [int(pq.split("_")[1]) for pq in group.keys() if pq.startswith("p")]
    )
    return species


def _preload_particle_species(
    single_file: bool,
    s: int,
    quantities: list[str],
    coord_type: str,
    outsteps: list[int],
    times: list[float],
    steps: list[int],
    coord_replacements: dict[str, str],
    file: h5py.File | list[h5py.File],
):
    prtl_data = {}
    for q in [
        f"X1_{s}",
        f"X2_{s}",
        f"X3_{s}",
        f"U1_{s}",
        f"U2_{s}",
        f"U3_{s}",
        f"W_{s}",
    ]:
        if q[0] in ["X", "U"]:
            q_ = coord_replacements[q.split("_")[0]]
        else:
            q_ = q.split("_")[0]
        if "p" + q not in quantities:
            continue
        if q not in prtl_data.keys():
            prtl_data[q_] = []
        if single_file:
            assert isinstance(file, h5py.File)
            for step_k in outsteps:
                group = file[step_k]
                if isinstance(group, h5py.Group):
                    if "p" + q in group.keys():
                        prtl_data[q_].append(group["p" + q])
                    else:
                        prtl_data[q_].append(np.full_like(prtl_data[q_][-1], np.nan))
                else:
                    raise ValueError(f"Unexpected type {type(file[step_k])}")
        else:
            for f in file:
                assert isinstance(f, h5py.File)
                group = f["Step0"]
                if isinstance(group, h5py.Group):
                    if "p" + q in group.keys():
                        prtl_data[q_].append(group["p" + q])
                    else:
                        prtl_data[q_].append(np.full_like(prtl_data[q_][-1], np.nan))
                else:
                    raise ValueError(f"Unexpected type {type(group)}")
        prtl_data[q_] = _list_to_ragged(prtl_data[q_])
        prtl_data[q_] = from_array(list(prtl_data[q_]))
        prtl_data[q_] = xr.DataArray(
            prtl_data[q_],
            dims=["t", "id"],
            name=q_,
            coords={"t": times, "s": ("t", steps)},
        )
    if coord_type == "sph":
        prtl_data["x"] = (
            prtl_data[coord_replacements["X1"]]
            * np.sin(prtl_data[coord_replacements["X2"]])
            * np.cos(prtl_data[coord_replacements["X3"]])
        )
        prtl_data["y"] = (
            prtl_data[coord_replacements["X1"]]
            * np.sin(prtl_data[coord_replacements["X2"]])
            * np.sin(prtl_data[coord_replacements["X3"]])
        )
        prtl_data["z"] = prtl_data[coord_replacements["X1"]] * np.cos(
            prtl_data[coord_replacements["X2"]]
        )
    return xr.Dataset(prtl_data)


# spectra
def _read_spectra_species(first_step: str, file: h5py.File):
    group = file[first_step]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Unexpected type {type(group)}")
    species = np.unique(
        [int(pq.split("_")[1]) for pq in group.keys() if pq.startswith("sN")]
    )
    return species


def _read_spectra_bins(first_step: str, log_bins: bool, file: h5py.File):
    group = file[first_step]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Unexpected type {type(group)}")
    e_bins = group["sEbn"]
    if not isinstance(e_bins, h5py.Dataset):
        raise ValueError(f"Unexpected type {type(e_bins)}")
    if log_bins:
        e_bins = np.sqrt(e_bins[1:] * e_bins[:-1])
    else:
        e_bins = (e_bins[1:] + e_bins[:-1]) / 2
    return e_bins


def _preload_spectra(
    single_file: bool,
    sp: int,
    e_bins: np.ndarray,
    outsteps: list[int],
    times: list[float],
    steps: list[int],
    file: h5py.File | list[h5py.File],
):
    dask_arrays = []
    if single_file:
        assert isinstance(file, h5py.File)
        for st in outsteps:
            array = from_array(file[f"{st}/sN_{sp}"])
            dask_arrays.append(array)
    else:
        for f in file:
            assert isinstance(f, h5py.File)
            array = from_array(f[f"Step0/sN_{sp}"])
            dask_arrays.append(array)

    return xr.DataArray(
        stack(dask_arrays, axis=0),
        dims=["t", "e"],
        name=f"n_{sp}",
        coords={
            "t": times,
            "s": ("t", steps),
            "e": e_bins,
        },
    )
