from typing import Union
import pandas as pd


class Diagnostics:
    df: Union[pd.DataFrame, None]

    def __init__(self, path: str):
        import os
        import logging
        import re

        outfiles = [o for o in os.listdir(path) if o.endswith(".out")]
        if len(outfiles) == 0:
            logging.warning(f"No .out files found in {path}")
            self.df = None
        else:
            self.outfile = os.path.join(path, outfiles[0])

            data = {}

            with open(self.outfile, "r") as f:
                content = f.read()
                steps = re.findall(r"Step:\s+(\d+)\.+\[", content)
                times = re.findall(r"Time:\s+([\d.]+\d)\.+\[", content)
                substeps = re.findall(r"\s+([A-Za-z]+)\.+([\d.]+)\s+([mµn]?s)", content)
                species = re.findall(
                    r"\s+species\s+(\d+)\s+\(.+\)\.+([\deE+-.]+)(\s+\d+\%\s:\s\d+\%\s+)?([\deE+-.]+)?( : )?([\deE+-.]+)?",
                    content,
                )

                data["steps"] = []
                for step in steps:
                    data["steps"].append(int(step))

                data["times"] = []
                for time in times:
                    data["times"].append(float(time))

                assert len(data["steps"]) == len(
                    data["times"]
                ), "Number of steps and times do not match"

                data["substeps"] = {}
                for substep in substeps:
                    if substep[0] not in data["substeps"].keys():
                        data["substeps"][substep[0]] = []

                    def to_ns(value: float, unit: str) -> float:
                        if unit == "s":
                            return value * 1e9
                        elif unit == "ms":
                            return value * 1e6
                        elif unit == "µs":
                            return value * 1e3
                        elif unit == "ns":
                            return value
                        else:
                            raise ValueError(f"Unknown time unit: {unit}")

                    data["substeps"][substep[0]].append(
                        to_ns(float(substep[1]), substep[2])
                    )

                for key in data["substeps"].keys():
                    assert len(data["substeps"][key]) == len(
                        data["steps"]
                    ), f"Number of substep entries for {key} does not match number of steps"

                data["species"] = {}
                data["species_min"] = {}
                data["species_max"] = {}
                for specie in species:
                    if specie[0] not in data["species"].keys():
                        data["species"][specie[0]] = []
                        data["species_min"][specie[0]] = []
                        data["species_max"][specie[0]] = []
                    data["species"][specie[0]].append(int(float(specie[1])))
                    if len(specie) == 6 and specie[3] != "" and specie[5] != "":
                        data["species_min"][specie[0]].append(int(float(specie[3])))
                        data["species_max"][specie[0]].append(int(float(specie[5])))

                for key in data["species"].keys():
                    assert len(data["species"][key]) == len(
                        data["steps"]
                    ), f"Number of species entries for {key} does not match number of steps"
                    assert (len(data["species_min"][key]) == len(data["steps"])) or (
                        len(data["species_min"][key]) == 0
                    ), f"Number of species min entries for {key} does not match number of steps"
                    assert (len(data["species_max"][key]) == len(data["steps"])) or (
                        len(data["species_max"][key]) == 0
                    ), f"Number of species max entries for {key} does not match number of steps"

            self.df = pd.DataFrame(index=data["steps"])
            self.df["Step"] = data["steps"]
            self.df["Time"] = data["times"]
            for key in data["substeps"].keys():
                self.df[key] = data["substeps"][key]
            for key in data["species"].keys():
                self.df[f"species_{key}"] = data["species"][key]
                if (
                    len(data["species_min"][key]) > 0
                    and len(data["species_max"][key]) > 0
                ):
                    self.df[f"species_{key}_min"] = data["species_min"][key]
                    self.df[f"species_{key}_max"] = data["species_max"][key]

            del data
