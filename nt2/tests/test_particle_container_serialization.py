import pickle

import numpy as np

from nt2.containers.particle_dataset import ParticleDataset
from nt2.containers.particles import ParticleContainer


class _RestoringParticleContainer(ParticleContainer):
    def _read_particles(self):
        raise AssertionError("deserialization should not rescan particle files")

    def _read_column(self, step, colname):
        return np.array([], dtype=np.int64)


def test_particle_dataset_is_restored_after_serialization():
    container = _RestoringParticleContainer.__new__(_RestoringParticleContainer)
    container.__dict__["_ParticleContainer__particles_defined"] = True
    container.__dict__["_ParticleContainer__particles"] = ParticleDataset(
        species=[1],
        steps=np.array([10]),
        times=np.array([2.5]),
        colnames=["x", "id", "sp"],
        read_column=container._read_column,
        partition_lengths=(0,),
    )

    restored = pickle.loads(pickle.dumps(container))

    assert restored.particles_defined
    assert restored.particles is not None
    assert restored.particles.species == [1]
    assert restored.particles.times.tolist() == [2.5]
