from dask.base import tokenize

from nt2.containers.container import BaseContainer
from nt2.readers.base import BaseReader
from nt2.utils import Format


class _Reader(BaseReader):
    @property
    def format(self) -> Format:
        return Format.HDF5


def test_base_container_has_deterministic_dask_token():
    container = BaseContainer(path="/tmp/sim", reader=_Reader(), remap=None)

    token1 = tokenize(container)
    token2 = tokenize(container)

    assert token1 == token2
