import os
from pathlib import Path

import pandas as pd
import pytest

from nt2.containers.diagnostics import Diagnostics


def _write_log(path: Path, steps: int = 3) -> None:
    preamble = "Entity diagnostic output\n\n"
    records = []
    for step in range(1, steps + 1):
        records.append(
            f"""Step: {step}....................[of {steps}]
Time: {step * 0.25:.2f}................[Δt = 0.25]

[SUBSTEP]..................[DURATION]
  Communications..............{step}.00 ms      3%
  FieldSolver................{step + 1}.00 µs      1%
  Injector.....................0.00 ns      0%

[PARTICLE SPECIES]            [TOTAL] [% TOT]
  species  1 (e-)............{step}.20e+03      1%
  species  2 (i+)............2.00e+03      1%

................................................................................

"""
        )
    path.write_text(preamble + "".join(records), encoding="utf-8")


def test_streaming_parser_without_cache(tmp_path: Path) -> None:
    outfile = tmp_path / "simulation.out"
    _write_log(outfile)

    diagnostics = Diagnostics(tmp_path, cache=False, chunk_size=2)

    expected = pd.DataFrame(
        {
            "Step": [1, 2, 3],
            "Time": [0.25, 0.50, 0.75],
            "Communications": [1e6, 2e6, 3e6],
            "FieldSolver": [2e3, 3e3, 4e3],
            "Injector": [0.0, 0.0, 0.0],
            "species_1": [1200, 2200, 3200],
            "species_2": [2000, 2000, 2000],
        }
    ).set_index("Step", drop=False)
    pd.testing.assert_frame_equal(diagnostics.df, expected)


def test_parquet_cache_is_reused_and_invalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outfile = tmp_path / "simulation.out"
    cache = tmp_path / "diagnostics.parquet"
    _write_log(outfile, steps=2)
    first = Diagnostics(outfile, cache_path=cache, chunk_size=1)
    assert first.df is not None
    assert cache.is_file()
    assert Path(f"{cache}.json").is_file()

    def fail_if_parsed(source: Path):
        raise AssertionError(f"Unexpected parse of {source}")

    monkeypatch.setattr(Diagnostics, "_records", fail_if_parsed)
    cached = Diagnostics(outfile, cache_path=cache)
    pd.testing.assert_frame_equal(cached.df, first.df)

    monkeypatch.undo()
    with outfile.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    os.utime(outfile, None)
    reparsed = Diagnostics(outfile, cache_path=cache)
    pd.testing.assert_frame_equal(reparsed.df, first.df)


def test_inconsistent_record_raises(tmp_path: Path) -> None:
    outfile = tmp_path / "simulation.out"
    _write_log(outfile, steps=2)
    text = outfile.read_text(encoding="utf-8")
    outfile.write_text(
        text.replace("  Injector.....................0.00 ns      0%\n", "", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Inconsistent diagnostics record at step 2"):
        Diagnostics(outfile, cache=False)


def test_incomplete_final_record_is_ignored(tmp_path: Path) -> None:
    outfile = tmp_path / "simulation.out"
    _write_log(outfile, steps=2)
    with outfile.open("a", encoding="utf-8") as stream:
        stream.write("Step: 3....................[of 3]\nTime: 0.75")

    diagnostics = Diagnostics(outfile, cache=False)

    assert diagnostics.df is not None
    assert diagnostics.df["Step"].tolist() == [1, 2]


def test_no_outfile(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    diagnostics = Diagnostics(tmp_path)

    assert diagnostics.df is None
    assert "No .out files found" in caplog.text


def test_cache_cannot_overwrite_source(tmp_path: Path) -> None:
    outfile = tmp_path / "simulation.out"
    _write_log(outfile)

    with pytest.raises(ValueError, match="must not overwrite"):
        Diagnostics(outfile, cache_path=outfile)
