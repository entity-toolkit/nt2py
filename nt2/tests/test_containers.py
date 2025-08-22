import pytest

from nt2.readers.base import BaseReader
from nt2.containers.fields import Fields
from nt2.containers.particles import Particles
from nt2.containers.data import Data
from nt2.tests.cases import TESTS


def check_shape(shape1, shape2):
    """
    Check if two shapes are equal
    """
    assert shape1 == shape2, f"Shape {shape1} is not equal to {shape2}"


@pytest.mark.parametrize(
    "test,field_container", [[test, fc] for test in TESTS for fc in [Data, Fields]]
)
def test_fields(test, field_container: type[Data] | type[Fields]):
    reader: BaseReader = test["reader"]()
    PATH = test["path"]
    if test["fields"] == {}:
        return

    coords: list[str] = ["x", "y", "z"]
    flds: list[str] = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]

    def coord_remap(Xold: str) -> str:
        return {
            "X1": "x",
            "X2": "y",
            "X3": "z",
        }.get(Xold, Xold)

    if test.get("coords", "cart") != "cart":
        coords = ["r", "th", "ph"]
        flds = ["Er", "Eth", "Eph", "Br", "Bth", "Bph"]
        coord_remap = lambda Xold: {
            "X1": "r",
            "X2": "th",
            "X3": "ph",
        }.get(Xold, Xold)

    def field_remap(Fold: str):
        return {
            f"f{F}{i+1}": f"{F}{x}" for i, x in enumerate(coords) for F in "EB"
        }.get(Fold, Fold)

    fields = field_container(
        path=PATH,
        reader=reader,
        remap={"coords": coord_remap, "fields": field_remap},
    )

    steps = reader.GetValidSteps(path=PATH, category="fields")
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
    "test,particle_container",
    [[test, fc] for test in TESTS for fc in [Data, Particles]],
)
def test_particles(test, particle_container: type[Data] | type[Particles]):
    reader: BaseReader = test["reader"]()
    PATH = test["path"]
    if test["particles"] == {}:
        return

    prtl_coords: list[str] = ["x", "y", "z", "ux", "uy", "uz", "w"]

    def prtl_remap(Xold: str) -> str:
        return {
            "pX1": "x",
            "pX2": "y",
            "pX3": "z",
            "pU1": "ux",
            "pU2": "uy",
            "pU3": "uz",
            "pW": "w",
        }.get(Xold, Xold)

    if test.get("coords", "cart") != "cart":
        prtl_coords = ["r", "th", "ph", "ur", "uth", "uph", "w"]
        prtl_remap = lambda Xold: {
            "pX1": "r",
            "pX2": "th",
            "pX3": "ph",
            "pU1": "ur",
            "pU2": "uth",
            "pU3": "uph",
            "pW": "w",
        }.get(Xold, Xold)
    particles = particle_container(
        path=PATH,
        reader=reader,
        remap={"particles": prtl_remap},
    )
    steps = reader.GetValidSteps(path=PATH, category="particles")
    assert particles.particles is not None, "Particles are None"
    for p in prtl_coords:
        if p == "z" and test["dim"] == "2D" and test.get("coords", "cart") == "cart":
            continue
        for i, (_, parts) in enumerate(particles.particles.items()):
            if isinstance(test["particles"]["num"], list):
                num = test["particles"]["num"][i]
            else:
                num = test["particles"]["num"]
            check_shape(parts[p].shape, (len(steps), num))
            for i, st in enumerate(steps):
                assert (
                    parts[p].isel(t=i).s.values[()] == st
                ), f"Step {st} does not match in particle {p}"
                check_shape(parts[p].isel(t=i).shape, (num,))
