import os
import tarfile
import pytest


@pytest.fixture(scope="session", autouse=True)
def unpack_testdata():
    base_dir = os.path.dirname(__file__)
    tar_path = os.path.join(base_dir, "testdata.tar.gz")
    extract_dir = os.path.join(base_dir, "testdata")

    if not os.path.exists(os.path.join(base_dir, "testdata")):
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir, filter="data")
