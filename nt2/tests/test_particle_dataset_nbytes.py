import numpy as np

from nt2.containers.particle_dataset import ParticleDataset


def _unexpected_read(step: int, column: str):
    raise AssertionError(f"nbytes unexpectedly read {column} at step {step}")


def test_nbytes_uses_cached_partition_lengths_without_computing():
    particles = ParticleDataset(
        species=[1],
        steps=np.array([10, 20], dtype=np.int64),
        times=np.array([1.0, 2.0], dtype=np.float64),
        colnames=["x", "id", "sp"],
        read_column=_unexpected_read,
        partition_lengths=(1, 2),
    )

    bytes_per_row = sum(
        np.dtype(dtype).itemsize
        for dtype in (np.int64, np.int32, np.int64, np.float64, np.int64)
    )
    assert particles.nbytes == 3 * bytes_per_row


def test_nbytes_tracks_timestep_partition_selection_without_computing():
    particles = ParticleDataset(
        species=[1],
        steps=np.array([10, 20], dtype=np.int64),
        times=np.array([1.0, 2.0], dtype=np.float64),
        colnames=["x", "id", "sp"],
        read_column=_unexpected_read,
        partition_lengths=(1, 2),
    )

    bytes_per_row = sum(
        np.dtype(dtype).itemsize
        for dtype in (np.int64, np.int32, np.int64, np.float64, np.int64)
    )
    assert particles.isel(t=-1).nbytes == 2 * bytes_per_row
