import pytest
from typer.testing import CliRunner
import matplotlib.pyplot as plt

import os
import nt2
from nt2.cli.main import app
from nt2.tests.cases import TESTS

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
    assert (
        nt2.__version__ in result.output
    ), f"Expected version {nt2.__version__} in output, got {result.output}"


@pytest.mark.parametrize(
    "test",
    [test for test in TESTS],
)
def test_show(test):
    PATH = test["path"]
    data = nt2.Data(PATH)
    result = runner.invoke(app, ["show", PATH])
    assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
    assert (
        data.to_str() in result.output
    ), f"Expected data info in output, got {result.output}"


@pytest.mark.parametrize(
    "test",
    [test for test in TESTS],
)
def test_plot_png(test):
    PATH = test["path"]
    if test["fields"] == {}:
        return
    if test.get("coords", "cart") == "cart":
        result = runner.invoke(
            app,
            [
                "plot",
                PATH,
                "--what",
                "fields",
                "--sel",
                "x=slice(None, 5);y=slice(-5.0, 5.0)",
                "--isel",
                f"t=0{';z=0' if test['dim'] == '3D' else ''}",
            ],
        )
        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"

        data = nt2.Data(PATH)
        fname = os.path.basename(PATH.strip("/"))

        d = data.fields.sel(x=slice(None, 5), y=slice(-5, 5)).isel(t=0)
        if test["dim"] == "3D":
            d = d.isel(z=0)
        d.inspect.plot(fig_kwargs={"dpi": 200})
        plt.savefig(fname=f"{fname}-2.png")

        def files_are_identical(path1, path2):
            with open(path1, "rb") as f1, open(path2, "rb") as f2:
                return f1.read() == f2.read()

        assert files_are_identical(
            f"{fname}-2.png", f"{fname}.png"
        ), f"Files {fname}-2.png and {fname}.png are not identical."

        os.remove(f"{fname}-2.png")
        os.remove(f"{fname}.png")
    # else:
    #     result = runner.invoke(
    #         app,
    #         [
    #             "plot",
    #             PATH,
    #             "--sel",
    #             "r=slice(None, 5);th=slice(1.5, 2.5)",
    #             "--isel",
    #             "t=0",
    #         ],
    #     )
