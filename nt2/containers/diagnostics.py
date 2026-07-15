"""Parse Entity diagnostic log files into pandas dataframes."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import pandas as pd


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_STEP_RE = re.compile(r"^Step:\s+(\d+)\.+\[")
_TIME_RE = re.compile(rf"^Time:\s+({_NUMBER})\.+\[")
_SUBSTEP_RE = re.compile(
    rf"^\s+(?P<name>[A-Za-z]+)\.+(?P<value>{_NUMBER})\s+"
    rf"(?P<unit>ms|[µμu]s|ns|s)\b"
)
_SPECIES_RE = re.compile(
    rf"^\s+species\s+(?P<species>\d+)\s+\([^)]*\)\.+"
    rf"(?P<total>{_NUMBER})"
    rf"(?:\s+\d+%\s*:\s*\d+%\s+(?P<minimum>{_NUMBER})"
    rf"\s*:\s*(?P<maximum>{_NUMBER}))?"
)

_UNIT_TO_NS = {
    "s": 1e9,
    "ms": 1e6,
    "µs": 1e3,
    "μs": 1e3,
    "us": 1e3,
    "ns": 1.0,
}


class Diagnostics:
    """Diagnostic data parsed from an Entity ``.out`` file.

    The source is read one line at a time and converted in bounded-size chunks.
    By default, the parsed dataframe is cached next to the source as Parquet.
    A valid cache is used on later construction instead of scanning the log.

    Parameters
    ----------
    path
        Directory containing a ``.out`` file, or the path to one ``.out`` file.
    cache
        Whether to read and write the Parquet cache.
    chunk_size
        Number of timestep records per Parquet row group.
    cache_path
        Optional cache filename. The default is
        ``<outfile>.diagnostics.parquet``.
    """

    _CACHE_VERSION = 1

    df: Optional[pd.DataFrame]
    outfile: str

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        cache: bool = True,
        chunk_size: int = 100_000,
        cache_path: Optional[Union[str, os.PathLike[str]]] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        outfile = self._find_outfile(Path(path))
        if outfile is None:
            logging.warning("No .out files found in %s", path)
            self.df = None
            return

        self.outfile = str(outfile)
        parquet_path = (
            Path(cache_path)
            if cache_path is not None
            else Path(f"{outfile}.diagnostics.parquet")
        )
        metadata_path = Path(f"{parquet_path}.json")
        if parquet_path.resolve() == outfile.resolve():
            raise ValueError("cache_path must not overwrite the source .out file")

        if cache and self._cache_is_valid(outfile, parquet_path, metadata_path):
            try:
                self.df = pd.read_parquet(parquet_path)
                return
            except (OSError, ValueError):
                logging.warning(
                    "Failed to read diagnostics cache %s; rebuilding it",
                    parquet_path,
                    exc_info=True,
                )

        if cache:
            self.df = self._parse_to_cache(
                outfile, parquet_path, metadata_path, chunk_size
            )
        else:
            chunks = list(self._dataframe_chunks(outfile, chunk_size))
            self.df = self._combine_chunks(chunks)

    @staticmethod
    def _find_outfile(path: Path) -> Optional[Path]:
        if path.is_file():
            if path.suffix != ".out":
                raise ValueError(f"Expected a .out file, got {path}")
            return path
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_dir():
            raise ValueError(f"Expected a directory or .out file, got {path}")

        outfiles = sorted(path.glob("*.out"))
        return outfiles[0] if outfiles else None

    @classmethod
    def _cache_is_valid(
        cls, source: Path, parquet_path: Path, metadata_path: Path
    ) -> bool:
        if not parquet_path.is_file() or not metadata_path.is_file():
            return False
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            stat = source.stat()
            return metadata == {
                "parser_version": cls._CACHE_VERSION,
                "source": str(source.resolve()),
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
            }
        except (OSError, ValueError, TypeError):
            return False

    @classmethod
    def _cache_metadata(cls, source: Path) -> Dict[str, Union[int, str]]:
        stat = source.stat()
        return {
            "parser_version": cls._CACHE_VERSION,
            "source": str(source.resolve()),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
        }

    @classmethod
    def _parse_to_cache(
        cls,
        source: Path,
        parquet_path: Path,
        metadata_path: Path,
        chunk_size: int,
    ) -> pd.DataFrame:
        # ParquetWriter creates multiple row groups in one file without retaining
        # the full parsed table in memory. Import lazily so cache=False only needs
        # pandas.
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_parquet = parquet_path.with_name(
            f".{parquet_path.name}.{os.getpid()}.tmp"
        )
        temporary_metadata = metadata_path.with_name(
            f".{metadata_path.name}.{os.getpid()}.tmp"
        )

        writer = None
        source_metadata = cls._cache_metadata(source)
        try:
            for chunk in cls._dataframe_chunks(source, chunk_size):
                table = pa.Table.from_pandas(chunk, preserve_index=True)
                if writer is None:
                    writer = pq.ParquetWriter(temporary_parquet, table.schema)
                elif table.schema != writer.schema:
                    raise ValueError(
                        "Diagnostics columns or dtypes changed within the log"
                    )
                writer.write_table(table)

            if writer is not None:
                writer.close()
                writer = None
            else:
                # Preserve the historical result for a log with no records.
                empty = pd.DataFrame(columns=["Step", "Time"])
                empty.index = pd.Index([], name="Step", dtype="int64")
                empty.to_parquet(temporary_parquet)

            with temporary_metadata.open("w", encoding="utf-8") as stream:
                # Use the source state from before parsing. If a running
                # simulation appended to the file meanwhile, this cache will
                # intentionally be stale and rebuilt on the next construction.
                json.dump(source_metadata, stream)

            os.replace(temporary_parquet, parquet_path)
            os.replace(temporary_metadata, metadata_path)
        finally:
            if writer is not None:
                writer.close()
            temporary_parquet.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)

        return pd.read_parquet(parquet_path)

    @classmethod
    def _dataframe_chunks(cls, source: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        records: List[Dict[str, Union[int, float]]] = []
        columns: Optional[List[str]] = None

        for record in cls._records(source):
            if columns is None:
                columns = list(record)
            else:
                missing = set(columns) - set(record)
                extra = set(record) - set(columns)
                if missing or extra:
                    raise ValueError(
                        f"Inconsistent diagnostics record at step {record['Step']}: "
                        f"missing={sorted(missing)}, extra={sorted(extra)}"
                    )
            records.append(record)
            if len(records) >= chunk_size:
                yield cls._make_dataframe(records, columns)
                records = []

        if records:
            # columns is necessarily set when records is non-empty.
            yield cls._make_dataframe(records, columns or [])

    @staticmethod
    def _make_dataframe(
        records: List[Dict[str, Union[int, float]]], columns: List[str]
    ) -> pd.DataFrame:
        dataframe = pd.DataFrame.from_records(records, columns=columns)
        return dataframe.set_index("Step", drop=False)

    @staticmethod
    def _combine_chunks(chunks: List[pd.DataFrame]) -> pd.DataFrame:
        if chunks:
            return pd.concat(chunks)
        dataframe = pd.DataFrame(columns=["Step", "Time"])
        dataframe.index = pd.Index([], name="Step", dtype="int64")
        return dataframe

    @staticmethod
    def _records(source: Path) -> Iterator[Dict[str, Union[int, float]]]:
        record: Optional[Dict[str, Union[int, float]]] = None

        with source.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                if line.startswith("Step:"):
                    if record is not None:
                        yield record
                    match = _STEP_RE.match(line)
                    record = {"Step": int(match.group(1))} if match else None
                    continue

                if record is None:
                    continue

                # Entity terminates each complete timestep with a dotted line.
                # Yielding here means a final record interrupted while the
                # simulation is still writing is not exposed as valid data.
                if line.startswith("........................................"):
                    yield record
                    record = None
                    continue

                if line.startswith("Time:"):
                    match = _TIME_RE.match(line)
                    if match:
                        record["Time"] = float(match.group(1))
                    continue

                if line.startswith("  species"):
                    match = _SPECIES_RE.match(line)
                    if match:
                        species = match.group("species")
                        record[f"species_{species}"] = int(float(match.group("total")))
                        minimum = match.group("minimum")
                        maximum = match.group("maximum")
                        if minimum is not None and maximum is not None:
                            record[f"species_{species}_min"] = int(float(minimum))
                            record[f"species_{species}_max"] = int(float(maximum))
                    continue

                if line.startswith("  "):
                    match = _SUBSTEP_RE.match(line)
                    if match:
                        record[match.group("name")] = (
                            float(match.group("value"))
                            * _UNIT_TO_NS[match.group("unit")]
                        )
