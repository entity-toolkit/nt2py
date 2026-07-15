from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from typer.testing import CliRunner

import os
import nt2
from nt2.cli.main import app
from nt2.tests.cases import TESTS

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
    assert nt2.__version__ in result.output, (
        f"Expected version {nt2.__version__} in output, got {result.output}"
    )


@pytest.mark.parametrize(
    "test",
    [test for test in TESTS],
)
def test_show(test, monkeypatch):
    path = test["path"]
    expected = f"Data summary for {path}"
    data = Mock()
    data.to_str.return_value = expected
    data_factory = Mock(return_value=data)
    monkeypatch.setattr(nt2, "Data", data_factory)

    result = runner.invoke(app, ["show", path])

    assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
    data_factory.assert_called_once_with(path)
    data.to_str.assert_called_once_with()
    assert expected in result.output


@pytest.mark.parametrize(
    "test",
    [test for test in TESTS if test["fields"]],
)
def test_plot_png(test, monkeypatch, tmp_path):
    path = test["path"]
    data_class = nt2.Data

    def field_data(data_path):
        return data_class(
            data_path,
            fields=True,
            particles=False,
            spectra=False,
            num_cpus=1,
        )

    monkeypatch.setattr(nt2, "Data", field_data)
    monkeypatch.chdir(tmp_path)

    is_cartesian = test.get("coords", "cart") == "cart"
    selection = (
        "x=slice(None, 5);y=slice(-5.0, 5.0)"
        if is_cartesian
        else "r=slice(None, 5);th=slice(1.5, 2.5)"
    )
    index_selection = f"t=0{';z=0' if test['dim'] == '3D' else ''}"
    result = runner.invoke(
        app,
        [
            "plot",
            path,
            "--what",
            "fields",
            "--sel",
            selection,
            "--isel",
            index_selection,
        ],
    )
    assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"

    fname = os.path.basename(path.strip("/"))
    actual_path = tmp_path / f"{fname}.png"
    expected_path = tmp_path / f"{fname}-expected.png"
    assert actual_path.is_file()

    data = field_data(path)
    if is_cartesian:
        selected = data.fields.sel(x=slice(None, 5), y=slice(-5, 5)).isel(t=0)
    else:
        selected = data.fields.sel(r=slice(None, 5), th=slice(1.5, 2.5)).isel(t=0)
    if test["dim"] == "3D":
        selected = selected.isel(z=0)

    plt.close("all")
    selected.inspect.plot(name=fname, fig_kwargs={"dpi": 200})
    plt.savefig(expected_path)
    plt.close("all")

    np.testing.assert_array_equal(
        plt.imread(actual_path),
        plt.imread(expected_path),
    )


@pytest.mark.parametrize("test", [test for test in TESTS if not test["fields"]])
def test_plot_without_fields_fails(test, monkeypatch, tmp_path):
    path = test["path"]
    data_class = nt2.Data

    def field_data(data_path):
        return data_class(
            data_path,
            fields=True,
            particles=False,
            spectra=False,
            num_cpus=1,
        )

    monkeypatch.setattr(nt2, "Data", field_data)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["plot", path, "--what", "fields", "--isel", "t=0"])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)
    assert str(result.exception).endswith("/fields'")
