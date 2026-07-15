from typing import List, Type, Union

import pytest

from nt2.readers.base import BaseReader
from nt2.containers.fields import FieldContainer
from nt2.containers.particles import ParticleContainer
from nt2.containers.data import Data
from nt2.tests.cases import TESTS


FIELD_TESTS = [test for test in TESTS if test["fields"]]
EMPTY_FIELD_TESTS = [test for test in TESTS if not test["fields"]]
PARTICLE_TESTS = [test for test in TESTS if test["particles"]]


def check_shape(shape1, shape2):
    """
    Check if two shapes are equal
    """
    assert shape1 == shape2, f"Shape {shape1} is not equal to {shape2}"


@pytest.mark.parametrize(
    "test,field_container",
    [[test, container] for test in FIELD_TESTS for container in [Data, FieldContainer]],
)
def test_fields(test, field_container: Union[Type[Data], Type[FieldContainer]]):
    reader: BaseReader = test["reader"]()
    path = test["path"]

    coords: List[str] = ["x", "y", "z"]
    flds: List[str] = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]

    def coord_remap_cart(Xold: str) -> str:
        return {
            "X1": "x",
            "X2": "y",
            "X3": "z",
        }.get(Xold, Xold)

    def coord_remap_sph(Xold: str) -> str:
        return {
            "X1": "r",
            "X2": "th",
            "X3": "ph",
        }.get(Xold, Xold)

    if test.get("coords", "cart") != "cart":
        coords = ["r", "th", "ph"]
        flds = ["Er", "Eth", "Eph", "Br", "Bth", "Bph"]

    def field_remap(Fold: str):
        return {
            f"f{F}{i + 1}": f"{F}{x}" for i, x in enumerate(coords) for F in "EB"
        }.get(Fold, Fold)

    remap = {
        "coords": (
            coord_remap_cart
            if test.get("coords", "cart") == "cart"
            else coord_remap_sph
        ),
        "fields": field_remap,
    }
    if field_container is Data:
        fields = field_container(
            path=path,
            fields=True,
            particles=False,
            spectra=False,
            reader=reader,
            remap=remap,
            num_cpus=1,
        )
    else:
        fields = field_container(
            path=path,
            reader=reader,
            remap=remap,
            coord_system=None,
            num_cpus=1,
        )

    steps = reader.GetValidSteps(path=path, category="fields", num_cpus=1)
    nx1 = test["fields"]["nx1"]
    nx2 = test["fields"]["nx2"]
    assert fields.fields is not None, "Fields are None"
    for f in flds:
        assert f in fields.fields, f"{f} is not in fields"
        if test["dim"] == "2D":
            xyzshape = (nx2, nx1)
            yzshape = (nx2,)
            xzshape = (nx1,)
            xyshape = ()
        else:
            nx3 = test["fields"]["nx3"]
            xyzshape = (nx3, nx2, nx1)
            yzshape = (nx3, nx2)
            xzshape = (nx3, nx1)
            xyshape = (nx2, nx1)

        check_shape(
            fields.fields[f].shape,
            tuple([len(steps), *xyzshape]),
        )
        check_shape(
            fields.fields[f].isel(t=0).shape,
            tuple([*xyzshape]),
        )
        if test.get("coords", "cart") == "cart":
            check_shape(
                fields.fields[f].isel(x=0).shape,
                tuple([len(steps), *yzshape]),
            )
            check_shape(
                fields.fields[f].isel(y=0).shape,
                tuple([len(steps), *xzshape]),
            )

            if test["dim"] == "3D":
                check_shape(
                    fields.fields[f].isel(z=0).shape,
                    tuple([len(steps), *xyshape]),
                )
        else:
            check_shape(
                fields.fields[f].isel(r=0).shape,
                tuple([len(steps), *yzshape]),
            )
            check_shape(
                fields.fields[f].isel(th=0).shape,
                tuple([len(steps), *xzshape]),
            )


@pytest.mark.parametrize(
    "test,field_container",
    [
        [test, container]
        for test in EMPTY_FIELD_TESTS
        for container in [Data, FieldContainer]
    ],
)
def test_missing_field_output_raises_file_not_found(
    test, field_container: Union[Type[Data], Type[FieldContainer]]
):
    reader: BaseReader = test["reader"]()
    kwargs = {
        "path": test["path"],
        "reader": reader,
        "remap": None,
        "num_cpus": 1,
    }

    with pytest.raises(FileNotFoundError, match="/fields"):
        if field_container is Data:
            field_container(
                **kwargs,
                fields=True,
                particles=False,
                spectra=False,
            )
        else:
            field_container(**kwargs, coord_system=None)


@pytest.mark.parametrize(
    "test,particle_container",
    [
        [test, container]
        for test in PARTICLE_TESTS
        for container in [Data, ParticleContainer]
    ],
)
def test_particles(
    test, particle_container: Union[Type[Data], Type[ParticleContainer]]
):
    reader: BaseReader = test["reader"]()
    path = test["path"]

    def prtl_remap_cart(Xold: str) -> str:
        return {
            "pX1": "x",
            "pX2": "y",
            "pX3": "z",
            "pU1": "ux",
            "pU2": "uy",
            "pU3": "uz",
            "pW": "w",
        }.get(Xold, Xold)

    def prtl_remap_sph(Xold: str) -> str:
        return {
            "pX1": "r",
            "pX2": "th",
            "pX3": "ph",
            "pU1": "ur",
            "pU2": "uth",
            "pU3": "uph",
            "pW": "w",
        }.get(Xold, Xold)

    remap = {
        "particles": (
            prtl_remap_cart if test.get("coords", "cart") == "cart" else prtl_remap_sph
        )
    }
    if particle_container is Data:
        particles = particle_container(
            path=path,
            fields=False,
            particles=True,
            spectra=False,
            reader=reader,
            remap=remap,
            num_cpus=1,
        )
    else:
        particles = particle_container(
            path=path,
            reader=reader,
            remap=remap,
            coord_system=None,
            num_cpus=1,
        )

    dataset = particles.particles
    assert dataset is not None, "Particles are None"
    assert len(dataset.species) == test["particles"].get("nspec", 4)

    selected = dataset.isel(t=-1).load(cols=["id", "sp", "w"])
    last_step = int(dataset.steps[-1])
    expected_counts = [
        reader.ReadArrayShapeAtTimestep(
            path=path,
            category="particles",
            quantity=f"pW_{species}",
            step=last_step,
        )[0]
        for species in dataset.species
    ]

    assert selected["st"].unique().tolist() == [last_step]
    assert selected.groupby("sp", sort=True).size().tolist() == expected_counts
    assert selected["w"].notna().all()
