import h5py
import numpy as np
import xarray as xr
from dask.array.core import from_array


from nt2.containers.container import Container
from nt2.containers.utils import _read_category_metadata_SingleFile


def _list_to_ragged(arr):
    max_len = np.max([len(a) for a in arr])
    return map(
        lambda a: np.concatenate([a, np.full(max_len - len(a), np.nan)]),
        arr,
    )


def _read_species_SingleFile(first_step: int, file: h5py.File):
    group = file[first_step]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Unexpected type {type(group)}")
    species = np.unique(
        [int(pq.split("_")[1]) for pq in group.keys() if pq.startswith("p")]
    )
    return species


def _preload_particle_species_SingleFile(
    s: int,
    quantities: list[str],
    coord_type: str,
    outsteps: list[int],
    times: list[float],
    steps: list[int],
    coord_replacements: dict[str, str],
    file: h5py.File,
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
        for step_k in outsteps:
            group = file[step_k]
            if isinstance(group, h5py.Group):
                if "p" + q in group.keys():
                    prtl_data[q_].append(group["p" + q])
                else:
                    prtl_data[q_].append(np.full_like(prtl_data[q_][-1], np.nan))
            else:
                raise ValueError(f"Unexpected type {type(file[step_k])}")
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


class ParticleContainer(Container):
    def __init__(self, **kwargs):
        super(ParticleContainer, self).__init__(**kwargs)
        PrtlDict = {
            "cart": {
                "X1": "x",
                "X2": "y",
                "X3": "z",
                "U1": "ux",
                "U2": "uy",
                "U3": "uz",
            },
            "sph": {
                "X1": "r",
                "X2": "θ" if self.configs["use_greek"] else "th",
                "X3": "φ" if self.configs["use_greek"] else "ph",
                "U1": "ur",
                "U2": "uΘ" if self.configs["use_greek"] else "uth",
                "U3": "uφ" if self.configs["use_greek"] else "uph",
            },
        }

        if self.configs["single_file"]:
            assert self.master_file is not None, "Master file not found"
            self.metadata["particles"] = _read_category_metadata_SingleFile(
                "p", self.master_file
            )
        self._particles = {}

        if len(self.metadata["particles"]["outsteps"]) > 0:
            if self.configs["single_file"]:
                assert self.master_file is not None, "Master file not found"
                species = _read_species_SingleFile(
                    self.metadata["particles"]["outsteps"][0], self.master_file
                )
                for s in species:
                    self._particles[s] = _preload_particle_species_SingleFile(
                        s=s,
                        quantities=self.metadata["particles"]["quantities"],
                        coord_type=self.configs["coordinates"],
                        outsteps=self.metadata["particles"]["outsteps"],
                        times=self.metadata["particles"]["times"],
                        steps=self.metadata["particles"]["steps"],
                        coord_replacements=PrtlDict[self.configs["coordinates"]],
                        file=self.master_file,
                    )

    @property
    def particles(self):
        return self._particles

    def print_particles(self) -> str:
        def sizeof_fmt(num, suffix="B"):
            for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
                if abs(num) < 1e3:
                    return f"{num:3.1f} {unit}{suffix}"
                num /= 1e3
            return f"{num:.1f} Y{suffix}"

        def compactify(lst):
            c = ""
            cntr = 0
            for l_ in lst:
                if cntr > 5:
                    c += "\n                "
                    cntr = 0
                c += l_ + ", "
                cntr += 1
            return c[:-2]

        string = ""
        if self.particles != {}:
            species = [int(i) for i in self.particles.keys()]
            string += "Particles:\n"
            string += f"  - species: {species}\n"
            string += f"  - data axes: {compactify(self.particles[species[0]].indexes.keys())}\n"
            string += f"  - timesteps: {self.particles[species[0]][list(self.particles[species[0]].data_vars.keys())[0]].shape[0]}\n"
            string += f"  - quantities: {compactify(self.particles[species[0]].data_vars.keys())}\n"
            size = 0
            for s in species:
                keys = list(self.particles[s].data_vars.keys())
                string += f"  - species [{s}]:\n"
                string += f"    - number: {self.particles[s][keys[0]].shape[1]}\n"
                size += self.particles[s].nbytes
            string += f"  - total size: {sizeof_fmt(size)}\n"
        else:
            string += "Particles: empty\n"
        return string
