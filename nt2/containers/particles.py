from typing import Any, Callable, List, Optional, Sequence, Tuple, Literal
from copy import copy

import dask.dataframe as dd
from dask.delayed import delayed
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.axes as maxes

from nt2.containers.container import BaseContainer


IntSelector = int | Sequence[int] | slice | Tuple[int, int]
FloatSelector = float | slice | Sequence[float] | Tuple[float, float]


class Selection:
    def __init__(
        self,
        type: Literal["value", "range", "list"],
        value: Optional[int | float | list | tuple] = None,
    ):
        self.type = type
        self.value = value

    def intersect(self, other: "Selection") -> "Selection":
        if self.value is None:
            return copy(other)
        elif other.value is None:
            return copy(self)
        if self.type == "value" and other.type == "value":
            if self.value == other.value:
                return Selection("value", self.value)
            else:
                return Selection("value")
        elif self.type == "value" and other.type == "list":
            assert isinstance(other.value, list), "other.value must be a list"
            if self.value in other.value:
                return Selection("value", self.value)
            else:
                return Selection("value")
        elif self.type == "value" and other.type == "range":
            assert (
                isinstance(other.value, tuple) and len(other.value) == 2
            ), "other.value must be a tuple of length 2"
            lo, hi = other.value
            if lo <= self.value < hi:
                return Selection("value", self.value)
            else:
                return Selection("value")
        elif self.type == "list" and other.type == "value":
            return other.intersect(self)
        elif self.type == "list" and other.type == "list":
            assert isinstance(self.value, list), "self.value must be a list"
            assert isinstance(other.value, list), "other.value must be a list"
            new_values = [v for v in self.value if v in other.value]
            return Selection("list", new_values)
        elif self.type == "list" and other.type == "range":
            assert (
                isinstance(other.value, tuple) and len(other.value) == 2
            ), "other.value must be a tuple of length 2"
            assert isinstance(self.value, list), "self.value must be a list"
            lo, hi = other.value
            new_values = [v for v in self.value if lo <= v <= hi]
            return Selection("list", new_values)
        elif self.type == "range" and other.type == "value":
            return other.intersect(self)
        elif self.type == "range" and other.type == "list":
            return other.intersect(self)
        elif self.type == "range" and other.type == "range":
            assert (
                isinstance(self.value, tuple) and len(self.value) == 2
            ), "self.value must be a tuple of length 2"
            assert (
                isinstance(other.value, tuple) and len(other.value) == 2
            ), "other.value must be a tuple of length 2"
            lo1, hi1 = self.value
            lo2, hi2 = other.value
            new_lo = max(lo1, lo2)
            new_hi = min(hi1, hi2)
            if new_lo <= new_hi:
                return Selection("range", (new_lo, new_hi))
            else:
                return Selection("value")
        else:
            raise ValueError(f"Unknown selection types: {self.type}, {other.type}")

    def __repr__(self) -> str:
        if self.type == "value":
            return "all" if self.value is None else f"{self.value:.3g}"
        elif self.type == "range":
            if self.value is None:
                return "all"
            else:
                assert (
                    isinstance(self.value, tuple) and len(self.value) == 2
                ), "value must be a tuple of length 2"
                lo, hi = self.value
                lo_str = "..." if lo is None or lo == -np.inf else f"{lo:.3g}"
                hi_str = "..." if hi is None or hi == np.inf else f"{hi:.3g}"
                return f"[ {lo_str} -> {hi_str} ]"
        elif self.type == "list":
            assert isinstance(self.value, list), "value must be a list"
            return "{ " + ", ".join(f"{v:.3g}" for v in self.value) + " }"
        else:
            return "InvalidSelection"

    def __str__(self) -> str:
        return self.__repr__()


def _coerce_selector_to_mask(
    s: IntSelector | FloatSelector,
    series: Any,
    inclusive_tuple: bool = True,
    method="exact",
):
    from operator import ior
    from functools import reduce

    if isinstance(s, slice):
        lo = s.start if s.start is not None else -np.inf
        hi = s.stop if s.stop is not None else np.inf
        step = s.step
        mask = (series >= lo) & (series <= hi)
        if step not in (None, 1):
            mask = mask & (((series - lo) % step) == 0)
        return mask, ("range", (lo, hi))
    elif isinstance(s, tuple) and len(s) == 2 and inclusive_tuple:
        lo, hi = s
        if lo is None:
            lo = -np.inf
        if hi is None:
            hi = np.inf
        return (series >= lo) & (series <= hi), ("range", (lo, hi))
    elif isinstance(s, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        if method == "exact":
            return series.isin(list(s)), ("list", list(s))
        else:
            return reduce(
                ior, [np.abs(series - v) == np.abs(series - v).min() for v in s]
            ), ("list", list(s))
    else:
        if method == "exact":
            return series == s, ("value", s)
        else:
            return np.abs(series - s) == np.abs(series - s).min(), ("value", s)


class ParticleDataset:
    def __init__(
        self,
        species: List[int],
        read_steps: Callable[[], np.ndarray],
        read_times: Callable[[], np.ndarray],
        read_column: Callable[[int, str], np.ndarray],
        read_colnames: Callable[[int], List[str]],
        fprec: Optional[type] = np.float32,
        selection: Optional[dict[str, Selection]] = None,
        ddf_index: dd.DataFrame | None = None,
    ):
        self.species = species
        self.read_steps = read_steps
        self.read_times = read_times
        self.read_column = read_column
        self.read_colnames = read_colnames
        self.fprec = fprec
        self.index_cols = ("id", "sp")
        self._all_columns_cache: Optional[List[str]] = None

        if selection is not None:
            self.selection = selection
        else:
            self.selection = {
                "t": Selection("range"),
                "st": Selection("range"),
                "sp": Selection("range"),
                "id": Selection("range"),
            }

        self.steps = read_steps()
        self.times = read_times()

        self._dtypes = {
            "id": np.int64,
            "sp": np.int32,
            "row": np.int64,
            "st": np.int64,
            "t": fprec,
            "x": fprec,
            "y": fprec,
            "z": fprec,
            "ux": fprec,
            "uy": fprec,
            "uz": fprec,
        }

        if ddf_index is not None:
            self._ddf_index = ddf_index
        else:
            self._ddf_index = self._build_index_ddf()

    @property
    def ddf(self) -> dd.DataFrame:
        return self._ddf_index

    @property
    def nbytes(self) -> int:
        return self.ddf.memory_usage(index=True, deep=True).sum().compute()

    @property
    def columns(self) -> List[str]:
        if self._all_columns_cache is None:
            self._all_columns_cache = self.read_colnames(self.steps[0])
        return self._all_columns_cache

    def sel(
        self,
        t: Optional[IntSelector | FloatSelector] = None,
        st: Optional[IntSelector] = None,
        sp: Optional[IntSelector] = None,
        id: Optional[IntSelector] = None,
        method: str = "exact",
    ) -> "ParticleDataset":
        ddf = self._ddf_index
        new_selection = {k: copy(v) for k, v in self.selection.items()}
        if st is not None:
            ddf_sel, (sel_type, sel_value) = _coerce_selector_to_mask(
                st, ddf["st"], method="exact"
            )
            ddf = ddf[ddf_sel]
            new_selection["st"] = new_selection["st"].intersect(
                Selection(sel_type, sel_value)
            )
        if t is not None:
            ddf_sel, (sel_type, sel_value) = _coerce_selector_to_mask(
                t, ddf["t"], method=method
            )
            ddf = ddf[ddf_sel]
            new_selection["t"] = new_selection["t"].intersect(
                Selection(sel_type, sel_value)
            )
        if sp is not None:
            ddf_sel, (sel_type, sel_value) = _coerce_selector_to_mask(
                sp, ddf["sp"], method="exact"
            )
            ddf = ddf[ddf_sel]
            new_selection["sp"] = new_selection["sp"].intersect(
                Selection(sel_type, sel_value)
            )
        if id is not None:
            ddf_sel, (sel_type, sel_value) = _coerce_selector_to_mask(
                id, ddf["id"], method="exact"
            )
            ddf = ddf[ddf_sel]
            new_selection["id"] = new_selection["id"].intersect(
                Selection(sel_type, sel_value)
            )

        return ParticleDataset(
            species=self.species,
            read_steps=self.read_steps,
            read_times=self.read_times,
            read_column=self.read_column,
            read_colnames=self.read_colnames,
            fprec=self.fprec,
            selection=new_selection,
            ddf_index=ddf,
        )

    def isel(
        self, t: Optional[IntSelector] = None, st: Optional[IntSelector] = None
    ) -> "ParticleDataset":
        ddf = self._ddf_index
        new_selection = {k: v for k, v in self.selection.items()}
        for t_or_s, t_or_s_str, t_or_s_arr in zip(
            [t, st], ["t", "st"], [self.times, self.steps]
        ):
            if t_or_s is not None:
                if isinstance(t_or_s, slice):
                    lo = t_or_s.start if t_or_s.start is not None else 0
                    hi = t_or_s.stop if t_or_s.stop is not None else -1
                    ddf_sel, (sel_type, sel_value) = _coerce_selector_to_mask(
                        slice(t_or_s_arr[lo], t_or_s_arr[hi]),
                        ddf[t_or_s_str],
                        method="exact",
                    )
                    ddf = ddf[ddf_sel]
                    new_selection[t_or_s_str] = new_selection[t_or_s_str].intersect(
                        Selection(sel_type, sel_value)
                    )
                elif isinstance(t_or_s, (list, tuple, np.ndarray, pd.Index, pd.Series)):
                    ddf_sel, (sel_type, sel_value) = _coerce_selector_to_mask(
                        [t_or_s_arr[ti] for ti in t_or_s],
                        ddf[t_or_s_str],
                        method="exact",
                    )
                    ddf = ddf[ddf_sel]
                    new_selection[t_or_s_str] = new_selection[t_or_s_str].intersect(
                        Selection(sel_type, sel_value)
                    )
                else:
                    ddf_sel, (sel_type, sel_value) = _coerce_selector_to_mask(
                        t_or_s_arr[t_or_s], ddf[t_or_s_str], method="exact"
                    )
                    ddf = ddf[ddf_sel]
                    new_selection[t_or_s_str] = new_selection[t_or_s_str].intersect(
                        Selection(sel_type, sel_value)
                    )
        return ParticleDataset(
            species=self.species,
            read_steps=self.read_steps,
            read_times=self.read_times,
            read_column=self.read_column,
            read_colnames=self.read_colnames,
            fprec=self.fprec,
            selection=new_selection,
            ddf_index=ddf,
        )

    def _build_index_ddf(self) -> dd.DataFrame:
        def _load_index_partition(st: int, t: float, index_cols: Tuple[str, ...]):
            cols = {c: self.read_column(st, c) for c in index_cols}
            n = len(next(iter(cols.values())))
            df = pd.DataFrame(cols)
            df["st"] = np.asarray(st, dtype=np.int64)
            df["t"] = np.asarray(t, dtype=float)
            df["row"] = np.arange(n, dtype=np.int64)
            return df

        delayed_parts = [
            delayed(_load_index_partition)(st, t, self.index_cols)
            for st, t in zip(self.steps, self.times)
        ]

        meta = pd.DataFrame(
            {
                **{
                    c: np.array([], dtype=self._dtypes.get(c, "O"))
                    for c in self.index_cols
                },
                "st": np.array([], dtype=self._dtypes.get("st", np.int64)),
                "t": np.array([], dtype=self._dtypes.get("t", np.int64)),
                "row": np.array([], dtype=self._dtypes.get("row", np.int64)),
            }
        )

        ddf = dd.from_delayed(delayed_parts, meta=meta)
        return ddf

    def load(self, cols: Sequence[str] | None = None) -> pd.DataFrame:
        if cols is None:
            cols = self.columns

        cols = [c for c in cols if c not in ("t", "st", "row")]

        meta_dict = {
            c: np.array([], dtype=self._dtypes.get(c, np.float64)) for c in cols
        }
        meta = self._ddf_index._meta.assign(**meta_dict)

        read_column = self.read_column
        cols_tuple = tuple(cols)

        def _attach_columns(part: pd.DataFrame) -> pd.DataFrame:
            if len(part) == 0:
                for c in cols_tuple:
                    part[c] = np.array([], dtype=meta.dtypes[c])
                return part
            st_val = int(part["st"].iloc[0])

            arrays = {c: read_column(st_val, c) for c in cols_tuple}

            sel = part["row"].to_numpy()
            for c in cols_tuple:
                part[c] = np.asarray(arrays[c])[sel]
            return part

        return (
            self._ddf_index.map_partitions(_attach_columns, meta=meta)
            .compute()
            .drop(columns=["row"])
        )

    def help(self) -> str:
        ret = "- use .sel(...) to select particles based on criteria:\n"
        ret += "  t  : time (float)\n"
        ret += "  st : step (int)\n"
        ret += "  sp : species (int)\n"
        ret += "  id : particle id (int)\n\n"
        ret += "  # example:\n"
        ret += "  #   .sel(t=slice(10.0, 20.0), sp=[1, 2, 3], id=[42, 22])\n\n"
        ret += "- use .isel(...) to select particles based on output step:\n"
        ret += "  t  : timestamp index (int)\n"
        ret += "  st : step index (int)\n\n"
        ret += "  # example:\n"
        ret += "  #   .isel(t=-1)\n"
        ret += "\n"
        ret += "- .sel and .isel can be chained together:\n\n"
        ret += "  # example:\n"
        ret += "  #   .isel(t=-1).sel(sp=1).sel(id=[55, 66])\n\n"
        ret += "- use .load(cols=[...]) to load data into a pandas DataFrame (`cols` defaults to all columns)\n\n"
        ret += "  # example:\n"
        ret += "  #  .sel(...).load()\n"
        return ret

    def __repr__(self) -> str:
        ret = "ParticleDataset:\n"
        ret += "================\n"
        ret += f"Variables:\n  {self.columns}\n\n"
        ret += "Current selection:\n"
        for k, v in self.selection.items():
            ret += f"  {k:<5} : {v}\n"
        ret += "\nHelp:\n"
        ret += "-----\n"
        ret += f"{self.help()}"
        return ret

    def __str__(self) -> str:
        return self.__repr__()

    def spectrum_plot(
        self,
        ax: maxes.Axes | None = None,
        bins: np.ndarray | None = None,
        quantity: Callable[[pd.DataFrame], np.ndarray] | None = None,
    ):
        if ax is None:
            ax = plt.gca()

        if "ux" in self.columns:
            cols = ["ux", "uy", "uz"]
            if quantity is None:
                uSqr = lambda df: np.sum(
                    [df[c].to_numpy(dtype=np.float64) ** 2 for c in cols], axis=0
                )
                quantity = lambda df: uSqr(df) * np.sqrt(1.0 + uSqr(df))
        else:
            cols = ["ur", "uth", "uph"]
            if quantity is None:
                uSqr = lambda df: np.sum(
                    [df[c].to_numpy(dtype=np.float64) ** 2 for c in cols], axis=0
                )
                quantity = lambda df: uSqr(df) * np.sqrt(1.0 + uSqr(df))
        df = self.load(cols=["sp", *cols])
        species = sorted(df["sp"].unique())
        arrays = df.groupby("sp").apply(quantity, include_groups=False)
        if bins is None:
            bins = np.logspace(0, 4, 100)
        hists = {sp: np.histogram(arrays[sp], bins=bins)[0] for sp in species}
        bins = 0.5 * (bins[1:] + bins[:-1])
        for sp in species:
            ax.loglog(bins, hists[sp], label=f"{sp}")
        if bins.min() > 0 and bins.max() / bins.min() > 100:
            ax.set(xscale="log", yscale="log")

    def phase_plot(
        self,
        ax: maxes.Axes | None = None,
        x_quantity: Callable[[pd.DataFrame], np.ndarray] | None = None,
        y_quantity: Callable[[pd.DataFrame], np.ndarray] | None = None,
        xy_bins: Tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs: Any,
    ):
        if ax is None:
            ax = plt.gca()

        if "ux" in self.columns:
            cols = ["ux", "uy", "uz"]
            for c in "xyz":
                if c in self.columns:
                    cols.append(c)
            if x_quantity is None:
                x_quantity = lambda df: df["x"].to_numpy(dtype=np.float64)
            if y_quantity is None:
                y_quantity = lambda df: df["ux"].to_numpy(dtype=np.float64)
        else:
            cols = ["ur", "uth", "uph"]
            for c in ["r", "th", "ph"]:
                if c in self.columns:
                    cols.append(c)
            if x_quantity is None:
                x_quantity = lambda df: df["r"].to_numpy(dtype=np.float64)
            if y_quantity is None:
                y_quantity = lambda df: df["ur"].to_numpy(dtype=np.float64)

        df = self.load(cols=[*cols])
        x_array = x_quantity(df)
        y_array = y_quantity(df)

        if xy_bins is None:
            x_bins = np.linspace(x_array.min(), x_array.max(), 100)
            y_bins = np.linspace(y_array.min(), y_array.max(), 100)
            xy_bins = (x_bins, y_bins)
        else:
            x_bins, y_bins = xy_bins

        h2d, xedges, yedges = np.histogram2d(x_array, y_array, bins=[x_bins, y_bins])
        X, Y = np.meshgrid(
            0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
        )
        pcm = ax.pcolormesh(
            X,
            Y,
            h2d.T,
            shading="auto",
            rasterized=True,
            **kwargs,
        )
        return pcm


class Particles(BaseContainer):
    """Parent class to manage the particles dataframe."""

    def __init__(self, **kwargs: Any) -> None:
        """Initializer for the Particles class.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the parent BaseContainer class.

        """
        super(Particles, self).__init__(**kwargs)
        if (
            self.reader.DefinesCategory(self.path, "particles")
            and self.particles_present
        ):
            self.__particles_defined = True
            self.__particles = self.__read_particles()
        else:
            self.__particles_defined = False
            self.__particles = None

    @property
    def particles_present(self) -> bool:
        """bool: Whether the particles are present in any of the timesteps."""
        return len(self.nonempty_steps) > 0

    @property
    def nonempty_steps(self) -> list[int]:
        """list[int]: List of timesteps that contain particles data."""
        valid_steps = self.reader.GetValidSteps(self.path, "particles")
        return [
            step
            for step in valid_steps
            if len(
                set(
                    q.split("_")[0]
                    for q in self.reader.ReadCategoryNamesAtTimestep(
                        self.path, "particles", "p", step
                    )
                    if q.startswith("p")
                )
            )
            > 0
        ]

    @property
    def particles_defined(self) -> bool:
        """bool: Whether the particles category is defined."""
        return self.__particles_defined

    @property
    def particles(self) -> ParticleDataset | None:
        """Returns the particles data.

        Returns
        -------
        ParticleDataset
            Dictionary of datasets for each step.

        """
        return self.__particles

    def __read_particles(self) -> ParticleDataset:
        """Helper function to read all particles data."""
        valid_steps = self.nonempty_steps

        quantities_ = [
            self.reader.ReadCategoryNamesAtTimestep(self.path, "particles", "p", step)
            for step in valid_steps
        ]
        quantities = sorted(np.unique([q for qtys in quantities_ for q in qtys]))

        unique_quantities = sorted(
            list(
                set(
                    str(q).split("_")[0]
                    for q in quantities
                    if not q.startswith("pIDX") and not q.startswith("pRNK")
                )
            )
        )
        all_species = sorted(list(set([int(str(q).split("_")[1]) for q in quantities])))

        sp_with_idx = sorted(
            [int(q.split("_")[1]) for q in quantities if q.startswith("pIDX")]
        )
        sp_without_idx = sorted([sp for sp in all_species if sp not in sp_with_idx])

        def remap_quantity(name: str) -> str:
            """
            Remaps the particle quantity name if remap is provided
            """
            if self.remap is not None and "particles" in self.remap:
                return self.remap["particles"](name)
            return name

        def GetCount(step: int, sp: int) -> np.int64:
            try:
                return np.int64(
                    self.reader.ReadArrayShapeAtTimestep(
                        self.path, "particles", f"pX1_{sp}", step
                    )[0]
                )
            except:
                return np.int64(0)

        def ReadSteps() -> np.ndarray:
            return np.array(self.reader.GetValidSteps(self.path, "particles"))

        def ReadTimes() -> np.ndarray:
            return self.reader.ReadPerTimestepVariable(
                self.path, "particles", "Time", "t"
            )["t"]

        def ReadColnames(step: int) -> list[str]:
            return [remap_quantity(q) for q in unique_quantities] + ["id", "sp"]

        def ReadColumn(step: int, colname: str) -> np.ndarray:
            read_colname = None
            if colname == "id":
                idx = np.concat(
                    [
                        self.reader.ReadArrayAtTimestep(
                            self.path, "particles", f"pIDX_{sp}", step
                        ).astype(np.int64)
                        for sp in sp_with_idx
                    ]
                    + [
                        np.zeros(GetCount(step, sp), dtype=np.int64) - 100
                        for sp in sp_without_idx
                    ]
                )
                if len(sp_with_idx) > 0 and f"pRNK_{sp_with_idx[0]}" in quantities:
                    rnk = np.concat(
                        [
                            self.reader.ReadArrayAtTimestep(
                                self.path, "particles", f"pRNK_{sp}", step
                            ).astype(np.int64)
                            for sp in sp_with_idx
                        ]
                        + [
                            np.zeros(GetCount(step, sp), dtype=np.int64) - 100
                            for sp in sp_without_idx
                        ]
                    )
                    return (idx + rnk) * (idx + rnk + 1) // 2 + rnk
                else:
                    return idx
            elif colname == "x" or colname == "r":
                read_colname = "pX1"
            elif colname == "y" or colname == "th":
                read_colname = "pX2"
            elif colname == "z" or colname == "ph":
                read_colname = "pX3"
            elif colname == "ux" or colname == "ur":
                read_colname = "pU1"
            elif colname == "uy" or colname == "uth":
                read_colname = "pU2"
            elif colname == "uz" or colname == "uph":
                read_colname = "pU3"
            elif colname == "w":
                read_colname = "pW"
            elif colname == "sp":
                return np.concat(
                    [
                        np.zeros(GetCount(step, sp), dtype=np.int32) + sp
                        for sp in sp_with_idx
                    ]
                    + [
                        np.zeros(GetCount(step, sp), dtype=np.int32) + sp
                        for sp in sp_without_idx
                    ]
                )
            else:
                read_colname = f"p{colname}"

            def species_has_quantity(sp: int) -> bool:
                return (
                    f"{read_colname}_{sp}"
                    in self.reader.ReadCategoryNamesAtTimestep(
                        self.path, "particles", "p", step
                    )
                )

            def get_quantity_for_species(sp: int) -> np.ndarray:
                if f"{read_colname}_{sp}" in quantities:
                    return self.reader.ReadArrayAtTimestep(
                        self.path, "particles", f"{read_colname}_{sp}", step
                    )
                else:
                    return np.zeros(GetCount(step, sp)) * np.nan

            return np.concat(
                [
                    get_quantity_for_species(sp)
                    for sp in sp_with_idx
                    if species_has_quantity(sp)
                ]
                + [
                    get_quantity_for_species(sp)
                    for sp in sp_without_idx
                    if species_has_quantity(sp)
                ]
            )

        return ParticleDataset(
            species=all_species,
            read_steps=ReadSteps,
            read_times=ReadTimes,
            read_colnames=ReadColnames,
            read_column=ReadColumn,
        )
