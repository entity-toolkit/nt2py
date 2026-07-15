import numpy as np
from dask.base import tokenize

from nt2.containers.base import BaseContainer
from nt2.readers.base import BaseReader
from nt2.utils import Format


class _Reader(BaseReader):
    @property
    def format(self) -> Format:
        return Format.HDF5

    def GetValidFilesAndSteps(self, path, category, steprange=None, num_cpus=None):
        return (["fields.00000001.h5"], [1])

    def ReadPerTimestepVariables(
        self, path, category, varnames, newnames, valid_files
    ):
        return {"t": np.array([0.5]), "s": np.array([1])}


def test_base_container_has_deterministic_dask_token():
    container = BaseContainer(
        path="/tmp/sim",
        category="fields",
        reader=_Reader(),
        remap=None,
        coord_system=None,
        num_cpus=1,
    )

    token1 = tokenize(container)
    token2 = tokenize(container)

    assert token1 == token2
