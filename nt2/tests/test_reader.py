import numpy as np
from nt2.tests.cases import TESTS

from nt2.readers.base import BaseReader
from nt2.utils import Layout


def pytest_generate_tests(metafunc):
    if "test" in metafunc.fixturenames:
        metafunc.parametrize("test", TESTS)


def check_equal_arrays(arr1, arr2):
    if isinstance(arr1, set):
        assert len(arr1) == len(
            arr2
        ), f"Set lengths do not match: {len(arr1)} != {len(arr2)}"
        assert arr1 == arr2, f"Sets do not match: {arr1} != {arr2}"
    else:
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        assert (
            arr1.shape == arr2.shape
        ), f"Shapes do not match: {arr1.shape} != {arr2.shape}"
        assert np.all(np.isclose(arr1, arr2)), f"Arrays do not match: {arr1} != {arr2}"


def check_raises(func, exception):
    try:
        func()
    except exception:
        return
    except Exception as e:
        raise e
    raise AssertionError(f"{func} should raise {exception}")


def test_reader(test):
    PATH = test["path"]
    invalid_tstep = test["invalid_tstep"]
    reader: BaseReader = test["reader"]()

    # PARAMS["2D"]["dx"] = PARAMS["2D"]["sx1"] / PARAMS["2D"]["nx1"]
    # PARAMS["2D"]["x1min"] = -PARAMS["2D"]["sx1"] / 2 + PARAMS["2D"]["dx"] / 2
    # PARAMS["2D"]["x2min"] = -PARAMS["2D"]["sx2"] / 2 + PARAMS["2D"]["dx"] / 2
    # PARAMS["2D"]["dt"] = PARAMS["2D"]["dx"] / np.sqrt(2) / 2

    # PARAMS["3D"]["dx"] = PARAMS["3D"]["sx1"] / PARAMS["3D"]["nx1"]
    # PARAMS["3D"]["x1min"] = -PARAMS["3D"]["sx1"] / 2 + PARAMS["3D"]["dx"] / 2
    # PARAMS["3D"]["x2min"] = -PARAMS["3D"]["sx2"] / 2 + PARAMS["3D"]["dx"] / 2
    # PARAMS["3D"]["x3min"] = PARAMS["3D"]["dx"] / 2
    # PARAMS["3D"]["dt"] = PARAMS["3D"]["dx"] / np.sqrt(3) / 2

    # dt = PARAMS[test["dim"]]["dt"]
    # dx = PARAMS[test["dim"]]["dx"]
    # x1min = PARAMS[test["dim"]]["x1min"]
    # x2min = PARAMS[test["dim"]]["x2min"]
    # nx1 = PARAMS[test["dim"]]["nx1"]
    # nx2 = PARAMS[test["dim"]]["nx2"]

    if test["fields"] != {}:
        # parameters
        nx1 = test["fields"]["nx1"]
        nx2 = test["fields"]["nx2"]
        sx1 = test["fields"]["sx1"]
        sx2 = test["fields"]["sx2"]
        dx = sx1 / nx1
        x1min = -sx1 / 2 + dx / 2
        x2min = -sx2 / 2 + dx / 2
        if test["dim"] == "3D":
            dt = dx / np.sqrt(3) / 2
        else:
            dt = dx / np.sqrt(2) / 2

        timestep = test.get("dt", None)
        if timestep is not None:
            dt = timestep

        field_names = test["fields"].get(
            "quantities",
            [f"{f}{i+1}" for i in range(3) for f in "BE"]
            + [f"N_{i}" for i in ["1_2", "3_4"]]
            + [f"T0{c+1}_{i+1}" for i in range(4) for c in range(3)],
        )
        field_names = set(f"f{f}" for f in field_names)
        # Check that invalid_tstep raises OSError in fields
        if invalid_tstep is not None:
            check_raises(
                lambda: reader.ReadArrayAtTimestep(
                    path=PATH, category="fields", quantity="Foo", step=invalid_tstep
                ),
                OSError,
            )

        # Check that timesteps are read correctly from fields
        times = reader.ReadPerTimestepVariable(
            path=PATH, category="fields", varname="Time", newname="t"
        )["t"]
        steps = reader.ReadPerTimestepVariable(
            path=PATH, category="fields", varname="Step", newname="s"
        )["s"]
        check_equal_arrays(
            times,
            np.array([s * dt for s in steps]),
        )

        # Check that the names of the fields are read correctly
        names = reader.ReadCategoryNamesAtTimestep(
            path=PATH, category="fields", prefix="f", step=1
        )
        check_equal_arrays(names, field_names)

        # Check coords
        if test.get("coords", "cart") == "cart":
            coords = reader.ReadFieldCoordsAtTimestep(path=PATH, step=1)
            x1 = np.array([x1min + i * dx for i in range(int(nx1))])
            x2 = np.array([x2min + i * dx for i in range(int(nx2))])
            check_equal_arrays(coords["X1"], x1)
            check_equal_arrays(coords["X2"], x2)

            if test["dim"] == "3D":
                sx3 = test["fields"]["sx3"]
                x3min = dx / 2
                nx3 = test["fields"]["nx3"]
                x3 = np.array([x3min + i * dx for i in range(int(nx3))])
                check_equal_arrays(coords["X3"], x3)

        # Check field shapes

        field = next(iter(field_names))
        layout = reader.ReadFieldLayoutAtTimestep(path=PATH, step=1)
        shape = reader.ReadArrayShapeAtTimestep(
            path=PATH, category="fields", quantity=field, step=1
        )
        if test["dim"] == "2D":
            check_equal_arrays(shape, (nx1, nx2) if layout == Layout.R else (nx2, nx1))
        else:
            nx3 = test["fields"]["nx3"]
            check_equal_arrays(
                shape, (nx1, nx2, nx3) if layout == Layout.R else (nx3, nx2, nx1)
            )

        for step in reader.GetValidSteps(path=PATH, category="fields"):
            for f in field_names:
                field = reader.ReadArrayAtTimestep(
                    path=PATH, category="fields", quantity=f, step=step
                )
                check_equal_arrays(field.shape, shape)

        reader.VerifySameCategoryNames(path=PATH, category="fields", prefix="f")
        reader.VerifySameFieldLayouts(path=PATH)

        # Check that the shapes of the fields are read correctly
        reader.VerifySameFieldShapes(path=PATH)

    if test["particles"] != {}:
        dt = 0
        if test["fields"] != {}:
            nx1 = test["fields"]["nx1"]
            sx1 = test["fields"]["sx1"]
            dx = sx1 / nx1
            if test["dim"] == "3D":
                dt = dx / np.sqrt(3) / 2
            else:
                dt = dx / np.sqrt(2) / 2

        timestep = test.get("dt", None)
        if timestep is not None:
            dt = timestep

        nspec: int = test["particles"].get("nspec", 4)

        prtl_names = (
            [f"U{i+1}_{j+1}" for i in range(3) for j in range(nspec)]
            + [
                f"X{i+1}_{j+1}"
                for i in range(
                    2
                    if test["dim"] == "2D" and test.get("coords", "cart") == "cart"
                    else 3
                )
                for j in range(nspec)
            ]
            + [f"W_{i+1}" for i in range(nspec)]
        )
        prtl_names = set(f"p{p}" for p in prtl_names)

        # Check that timesteps are read correctly from particles
        times = reader.ReadPerTimestepVariable(
            path=PATH, category="particles", varname="Time", newname="t"
        )["t"]
        steps = reader.ReadPerTimestepVariable(
            path=PATH, category="particles", varname="Step", newname="s"
        )["s"]

        if dt is not None:
            check_equal_arrays(
                times,
                np.array([s * dt for s in steps]),
            )

        # Check that the names of the particle quantities are read correctly
        names = reader.ReadCategoryNamesAtTimestep(
            path=PATH, category="particles", prefix="p", step=1
        )
        print(names, prtl_names)
        check_equal_arrays(names, prtl_names)

        # Check prtl shapes
        for step in reader.GetValidSteps(path=PATH, category="particles"):
            for sp in range(nspec):
                shape = reader.ReadArrayShapeAtTimestep(
                    path=PATH,
                    category="particles",
                    quantity=f"pW_{sp+1}",
                    step=step,
                )
                for p in prtl_names:
                    if not p.endswith(f"_{sp+1}"):
                        continue
                    prtl_shape = reader.ReadArrayShapeAtTimestep(
                        path=PATH, category="particles", quantity=p, step=step
                    )
                    check_equal_arrays(prtl_shape, shape)

        # Check that all timesteps have the same names
        reader.VerifySameCategoryNames(path=PATH, category="particles", prefix="p")
