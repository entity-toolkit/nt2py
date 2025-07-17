import typer, nt2, os
from typing_extensions import Annotated
import matplotlib.pyplot as plt

app = typer.Typer()


@app.command(help="Print the data info")
def version():
    print(nt2.__version__)


def check_path(path: str) -> str:
    if not os.path.exists(path) or not (
        os.path.exists(os.path.join(path, "fields"))
        or os.path.exists(os.path.join(path, "particles"))
        or os.path.exists(os.path.join(path, "spectra"))
    ):
        raise typer.BadParameter(
            f"Path {path} does not exist or is not a valid nt2 data directory."
        )
    return path


def check_sel(sel: str) -> dict[str, int | float | slice]:
    if sel == "":
        return {}
    sel_list = sel.strip().split(";")
    sel_dict = {}
    for _, s in enumerate(sel_list):
        coord, arg = s.strip().split("=", 1)
        coord = coord.strip()
        arg_exec = eval(arg.strip())
        assert isinstance(
            arg_exec, (int, float, slice)
        ), f"Invalid selection argument for '{coord}': {arg_exec}. Must be int, float, or slice."
        sel_dict[coord] = arg_exec
    return sel_dict


def check_species(species: int) -> int:
    if species < 0:
        raise typer.BadParameter(
            f"Species index must be a non-negative integer, got {species}."
        )
    return species


def check_what(what: str) -> str:
    valid_options = ["fields", "particles", "spectra"]
    if what not in valid_options:
        raise typer.BadParameter(
            f"Invalid option '{what}'. Valid options are: {', '.join(valid_options)}."
        )
    return what


@app.command(help="Print the data info")
def show(
    path: Annotated[
        str,
        typer.Argument(
            callback=check_path,
            help="Path to the data",
        ),
    ] = "",
):
    data = nt2.Data(path)
    print(data.to_str())


@app.command(help="Plot the data")
def plot(
    path: Annotated[
        str,
        typer.Argument(
            callback=check_path,
            help="Path to the data",
        ),
    ] = "",
    what: Annotated[
        Annotated[
            str,
            typer.Option(
                callback=check_what,
                help="Which data to plot [fields, particles, spectra]",
            ),
        ],
        str,
    ] = "fields",
    fields: Annotated[
        str,
        typer.Option(
            help="Which fields to plot (only when `what` is `fields`). Separate multiple fields with ';'. May contain regex. Empty = all fields. Example: `--fields \"E.*;B.*\"`",
        ),
    ] = "",
    species: Annotated[
        Annotated[
            int,
            typer.Option(
                callback=check_species,
                help="Which species to take (only when `what` is `particles`). 0 = all species",
            ),
        ],
        str,
    ] = 0,
    sel: Annotated[
        str,
        typer.Option(
            callback=check_sel,
            help="Select a subset of the data with xarray.sel. Separate multiple selections with ';'. Example: `--sel \"t=23;z=slice(0, None)\"`",
        ),
    ] = "",
    isel: Annotated[
        str,
        typer.Option(
            callback=check_sel,
            help="Select a subset of the data with xarray.isel. Separate multiple selections with ';'. Example: `--isel \"t=slice(None, 5);z=5\"`",
        ),
    ] = "",
):
    fname = os.path.basename(path.strip("/"))
    data = nt2.Data(path)
    assert isinstance(
        sel, dict
    ), f"Invalid selection format: {sel}. Must be a dictionary."
    assert isinstance(isel, dict), f"Invalid isel format: {isel}. Must be a dictionary."
    if what == "fields":
        d = data.fields
        if sel != {}:
            slices = {}
            sels = {}
            slices = {k: v for k, v in sel.items() if isinstance(v, slice)}
            sels = {k: v for k, v in sel.items() if not isinstance(v, slice)}
            d = d.sel(**sels, method="nearest")
            d = d.sel(**slices)
        if isel != {}:
            d = d.isel(**isel)
        if fields != "":
            ret = d.inspect.plot(
                name=fname, only_fields=fields.split(";"), fig_kwargs={"dpi": 200}
            )
        else:
            ret = d.inspect.plot(name=fname, fig_kwargs={"dpi": 200})
        if not isinstance(ret, bool):
            plt.savefig(fname=f"{fname}.png")

    elif what == "particles":
        raise NotImplementedError("Particles plotting is not implemented yet.")
    elif what == "spectra":
        raise NotImplementedError("Spectra plotting is not implemented yet.")
    else:
        raise typer.BadParameter(
            f"Invalid option '{what}'. Valid options are: fields, particles, spectra."
        )
