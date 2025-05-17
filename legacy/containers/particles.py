import os
import h5py
from nt2.containers.container import Container
from nt2.containers.utils import (
    _read_category_metadata,
    _read_particle_species,
    _preload_particle_species,
)


class ParticleContainer(Container):
    """
    * * * * ParticleContainer : Container * * * *

    Class for holding the particle data.

    Attributes
    ----------
    particles : dict
        The dictionary of particle species.

    particle_files : list
        The list of opened particle files.

    Methods
    -------
    print_particles()
        Prints the basic information about the particle data.

    """

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
            self.metadata["particles"] = _read_category_metadata(
                True, "p", self.master_file
            )
        else:
            particle_path = os.path.join(self.path, "particles")
            if os.path.isdir(particle_path):
                files = sorted(os.listdir(particle_path))
                try:
                    self.particle_files = [
                        h5py.File(os.path.join(particle_path, f), "r") for f in files
                    ]
                except OSError:
                    raise OSError(f"Could not open file in {particle_path}")
                self.metadata["particles"] = _read_category_metadata(
                    False, "p", self.particle_files
                )
        self._particles = {}

        if (
            "particles" in self.metadata
            and len(self.metadata["particles"]["outsteps"]) > 0
        ):
            if self.configs["single_file"]:
                assert self.master_file is not None, "Master file not found"
                species = _read_particle_species(
                    self.metadata["particles"]["outsteps"][0], self.master_file
                )
            else:
                species = _read_particle_species("Step0", self.particle_files[0])
            self.metadata["particles"]["species"] = species
            for s in species:
                self._particles[s] = _preload_particle_species(
                    self.configs["single_file"],
                    s=s,
                    quantities=self.metadata["particles"]["quantities"],
                    coord_type=self.configs["coordinates"],
                    outsteps=self.metadata["particles"]["outsteps"],
                    times=self.metadata["particles"]["times"],
                    steps=self.metadata["particles"]["steps"],
                    coord_replacements=PrtlDict[self.configs["coordinates"]],
                    file=(
                        self.master_file
                        if self.configs["single_file"] and self.master_file is not None
                        else self.particle_files
                    ),
                )

    @property
    def particles(self):
        return self._particles

    def __del__(self):
        if not self.configs["single_file"]:
            for f in self.particle_files:
                f.close()

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
