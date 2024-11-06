import h5py
import numpy as np


def _read_category_metadata_SingleFile(prefix: str, file: h5py.File):
    f_outsteps = []
    f_steps = []
    f_times = []
    f_quantities = None
    for st in file:
        group = file[st]
        if isinstance(group, h5py.Group):
            if any([k.startswith(prefix) for k in group if k is not None]):
                if f_quantities is None:
                    f_quantities = [k for k in group.keys() if k.startswith(prefix)]
                f_outsteps.append(st)
                time_ds = group["Time"]
                if isinstance(time_ds, h5py.Dataset):
                    f_times.append(time_ds[()])
                else:
                    raise ValueError(f"Unexpected type {type(time_ds)}")
                step_ds = group["Step"]
                if isinstance(step_ds, h5py.Dataset):
                    f_steps.append(int(step_ds[()]))
                else:
                    raise ValueError(f"Unexpected type {type(step_ds)}")

        else:
            raise ValueError(f"Unexpected type {type(file[st])}")
    f_outsteps = sorted(f_outsteps, key=lambda x: int(x.replace("Step", "")))
    f_steps = sorted(f_steps)
    f_times = np.array(sorted(f_times), dtype=np.float64)
    return {
        "quantities": f_quantities,
        "outsteps": f_outsteps,
        "steps": f_steps,
        "times": f_times,
    }
