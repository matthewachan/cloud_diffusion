import os
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pandas as pd
from herbie import FastHerbie

from cloud_diffusion.utils import parse_args

config = SimpleNamespace(
    dl_dir="/fs/nexus-scratch/mattchan/datasets/hrrr",
    start_date="2018-01-01",
    end_date="2023-01-01",
    freq="4H",
    n_cpu=16,
    region="west",  # US west coast
)


if __name__ == "__main__":
    parse_args(config)

    # Get datetimes in 4-hour intervals
    datetimes = pd.date_range(
        start=config.start_date, end=config.end_date, freq=config.freq
    )
    # Ensure that the number of datetimes is divisible by 4
    datetimes = datetimes[: 4 * (len(datetimes) // 4)]
    datetimes = np.array(datetimes, dtype=np.datetime64)
    # Reshape into time windows of 4 datetimes (the first 3 are inputs, the last is the target)
    time_windows = datetimes.reshape((len(datetimes) // 4, 4))
    print(time_windows.shape)

    def save_data(date_tuple):
        """Download and save temperature data for a given time window"""
        i, time_window = date_tuple
        time_window = pd.to_datetime(time_window)
        start_date = time_window[0].strftime("%Y-%m-%d")
        ds = FastHerbie(
            time_window, model="hrrr", product="sfc", save_dir=config.dl_dir
        ).xarray("TMP:2 m")
        if len(ds.t2m) != 4:
            print(f"Skipping {start_date}")
            return
        # Crop to 1:1 aspect ratio around a region in the US
        if config.region == "west":
            temps = ds.t2m.values[:, :1024, :1024]
        elif config.region == "east":
            temps = ds.t2m.values[:, :1024, -1024:]
        # Saves temperature data as a .npy file
        np.save(os.path.join(config.dl_dir, f"sample_{i}.npy"), temps)

    p = Pool(config.n_cpu)
    p.map(save_data, enumerate(time_windows))
    p.close()
