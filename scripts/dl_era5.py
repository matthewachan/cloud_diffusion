import datetime
import os
from multiprocessing import Pool

import cdsapi
import pandas as pd

# Define multiprocessing function
def download_era5(date):
    year = str(date.year)
    month = str(date.month)
    day = str(date.day)
    time = f"{date.hour:02d}:{date.minute:02d}"
    try:
        c.retrieve(
            "reanalysis-era5-land",
            {
                "variable": [
                    # "10m_u_component_of_wind",
                    # "10m_v_component_of_wind",
                    "2m_temperature",
                ],
                "year": year,
                "month": month,
                "day": day,
                "time": time,
                "area": [65, -126, 10, -70],  # North, West, South, East. Default: global
            },
            f"/fs/nexus-scratch/mattchan/datasets/era5-land/{year}_{month}_{day}_{time}.grib2",
        )
    except:
        print(f"Failed to download {year}_{month}_{day}_{time}.grib2")


# start_date = datetime.datetime(2010, 1, 1, 0, 0)
# end_date = datetime.datetime(2020, 1, 1, 0, 0)
# date_range = pd.date_range(start=start_date, end=end_date, freq='3H')
# c = cdsapi.Client(quiet=True)
# p = Pool(16)
# p.map(download_era5, date_range)
# p.close()

import xarray as xr
import numpy as np

def convert_to_npy(date):
    year = str(date.year)
    month = str(date.month)
    day = str(date.day)
    time = f"{date.hour:02d}:{date.minute:02d}"
    try:
        temp = xr.open_dataset(
            f"/fs/nexus-scratch/mattchan/datasets/era5-land/{year}_{month}_{day}_{time}.grib2",
            engine="cfgrib",
        )["t2m"].values
        np.save(
            f"/fs/nexus-scratch/mattchan/datasets/era5-land/{year}_{month}_{day}_{time}.npy",
            temp,
        )
    except:
        print(f"Failed to convert {year}_{month}_{day}_{time}.grib2")

if __name__ == "__main__":
    # start = time.time()
    # dataset = Era5Land('/fs/nexus-scratch/mattchan/datasets/era5-land')
    # loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)
    # for batch in tqdm.tqdm(loader, total=len(loader)):
    #     a = batch
    # print(time.time() - start)
    start_date = datetime.datetime(2010, 1, 1, 0, 0)
    end_date = datetime.datetime(2020, 1, 1, 0, 0)
    date_range = pd.date_range(start=start_date, end=end_date, freq='3H')
    p = Pool(16)
    p.map(convert_to_npy, date_range)
    p.close()
    # for date in tqdm.tqdm(dataset.date_range, total=len(dataset.date_range)):
    #     filename = dataset.date_to_filename(date)
    #     assert os.path.exists(os.path.join(dataset.dir, filename)), f"Missing {filename}"
    #     temp = xr.open_dataset(os.path.join(dataset.dir, filename), engine='cfgrib')['t2m'].values
    #     np.save(os.path.join(dataset.dir, filename.replace('.grib2', '.npy')), temp)