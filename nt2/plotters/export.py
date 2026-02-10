from typing import Any, Callable, Union, Optional
import matplotlib.pyplot as plt


def makeFramesAndMovie(
    name: str,
    plot: Callable,
    times: list[float],
    data: Any = None,
    **kwargs: Any,
) -> bool:
    num_cpus = kwargs.pop("num_cpus", None)
    if all(
        makeFrames(
            plot=plot,
            times=times,
            fpath=f"{name}/frames",
            data=data,
            num_cpus=num_cpus,
        )
    ):
        print(f"Frames saved in {name}/frames")
        output: str = kwargs.pop("output", f"{name}.mp4")
        if makeMovie(
            input=f"{name}/frames/",
            overwrite=True,
            output=output,
            number=5,
            **kwargs,
        ):
            print(f"Movie {name}.mp4 created successfully")
            return True
        else:
            return False
    else:
        raise ValueError("Failed to make frames")


def makeMovie(**ffmpeg_kwargs: Union[str, int, float]) -> bool:
    """
    Create a movie from frames using the `ffmpeg` command-line tool.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments for the `ffmpeg` command-line tool.

    Returns
    -------
    bool
        True if the movie was created successfully, False otherwise.

    Notes
    -----
    This function uses the `subprocess` module to execute the `ffmpeg` command-line
    tool with the given arguments.

    Examples
    --------
    >>> makeMovie(ffmpeg="/path/to/ffmpeg", framerate=30, start=0, input="step_", number=3,
                  extension="png", compression=1, overwrite=True, output="anim.mp4")
    """
    import subprocess

    input_pattern: str = (
        f"{ffmpeg_kwargs.get('input', 'step_')}%0{ffmpeg_kwargs.get('number', 3)}d.{ffmpeg_kwargs.get('extension', 'png')}"
    )

    command = [
        ffmpeg_kwargs.get("ffmpeg", "ffmpeg"),
        "-nostdin",
        "-framerate",
        str(ffmpeg_kwargs.get("framerate", 30)),
        "-start_number",
        str(ffmpeg_kwargs.get("start", 0)),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-crf",
        str(ffmpeg_kwargs.get("compression", 1)),
        "-filter_complex",
        "[0:v]format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-y" if ffmpeg_kwargs.get("overwrite", False) else None,
        ffmpeg_kwargs.get("output", "movie.mp4"),
    ]
    command = [str(c) for c in command if c is not None]
    print("Command:\n", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("ffmpeg -- [OK]")
        return True
    else:
        print("ffmpeg -- [not OK]", result.returncode, result.stdout, result.stderr)
        return False


def _plot_and_save(ti: int, t: float, fpath: str, plot: Callable, data: Any) -> bool:
    try:
        if data is None:
            plot(t)
        else:
            plot(t, data)
        plt.savefig(f"{fpath}/{ti:05d}.png")
        plt.close()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def makeFrames(
    plot: Callable,
    times: list[float],
    fpath: str,
    data: Any = None,
    num_cpus: Optional[int] = None,
) -> list[bool]:
    """
    Create plot frames from a set of timesteps of the same dataset.

    Parameters
    ----------
    plot : function
        A function that generates and saves the plot. The function must take a time index
        or a timestamp as an argument and, optionally, the data object.

    times : array_like, optional
        The time indices to use for generating the movie.
        Can either be timestep indices or timestamps.
        Must coincide with the time accepted by the `plot` function.

    fpath : str
        The file path to save the frames.

    data : xarray.Dataset, optional
        The dataset to use for generating the movie (passed to plot as the second argument)

    num_cpus : int, optional
        The number of CPUs to use for parallel processing. If None, use all available CPUs.

    Returns
    -------
    list
        A list of results returned by the `plot` function, one for each time index.

    Raises
    ------
    ValueError
        If `plot` is not a callable function.

    Notes
    -----
    This function uses the `multiprocessing` module to parallelize the generation
    of the plots, and `tqdm` module to display a progress bar.

    Examples
    --------
    >>> makeFrames(plot_func, range(100), 'output/', num_cpus=16)

    """
    from loky import get_reusable_executor
    from tqdm import tqdm
    import os

    os.makedirs(fpath, exist_ok=True)

    ex = get_reusable_executor(max_workers=num_cpus or (os.cpu_count() or 1))
    futures = [
        ex.submit(_plot_and_save, ti, t, fpath, plot, data)
        for ti, t in enumerate(times)
    ]
    return [
        f.result()
        for f in tqdm(
            futures,
            total=len(futures),
            desc=f"rendering frames to {fpath}",
            unit="frame",
        )
    ]
