import datetime
import glob
import os

import tqdm
import time
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import xarray as xr
from cloud_diffusion.utils import MAX_TEMP, MIN_TEMP, rescale


class TemperatureDataset(torch.utils.data.Dataset):
    def __init__(self, dir, img_size):
        self.transform = T.Resize((img_size, img_size), antialias=True)
        self.fnames = glob.glob(os.path.join(dir, "*.npy"))

    def __getitem__(self, index):
        sample = np.load(self.fnames[index])
        sample = 0.5 - rescale(sample, MIN_TEMP, MAX_TEMP)
        return self.transform(torch.from_numpy(sample))

    def __len__(self):
        return len(self.fnames)

class KelvinToFahrenheit(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return (sample - 273.15) * (9/5) + 32

# Normalize to [-1, 1]
class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return 2 * rescale(sample, MIN_TEMP, MAX_TEMP) - 1

# Strategy: All datetimes are valid (maybe pick a random subset of this). Pick a random starting datetime and a 4 frame sliding window around that datetime. 
class Era5Land(torch.utils.data.Dataset):
    def __init__(self, dir, img_size=512, window_size=4):
        self.dir = dir
        self.window_size = window_size
        self.img_size = img_size

        start_date = datetime.datetime(2010, 1, 1, 0, 0)
        end_date = datetime.datetime(2020, 1, 1, 0, 0) - 4 * datetime.timedelta(hours=3)
        self.date_range = pd.date_range(start=start_date, end=end_date, freq='3H')
        self.tfm = T.Compose([T.ToTensor(), T.CenterCrop([550, 550]), T.Resize([img_size, img_size], antialias=True), Normalize()])
    
    def __getitem__(self, index):
        temps = np.array([np.load(os.path.join(self.dir, self.date_to_filename(time))) for time in self.date_range[index:index+self.window_size]])
        temps = np.nan_to_num(temps, nan=MIN_TEMP)
        # Apply transform
        temps = self.tfm(np.stack(temps, axis=-1))
        return temps # Output shape [window_size, 550, 550]

    
    def __len__(self):
        return len(self.date_range)
    
    def date_to_filename(self, date):
        return f"{date.year}_{date.month}_{date.day}_{date.hour:02d}:{date.minute:02d}.npy"


# Iterate through all dates and verify that files exist (nothing missing)
if __name__ == "__main__":
    # start = time.time()
    dataset = Era5Land('/fs/nexus-scratch/mattchan/datasets/era5-land/npy')
    sample = dataset[0]
    print(sample.shape, sample.amin(), sample.amax())
    # loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)
    # for batch in tqdm.tqdm(loader, total=len(loader)):
    #     a = batch
    # print(time.time() - start)
    # for date in tqdm.tqdm(dataset.date_range, total=len(dataset.date_range)):
    #     filename = dataset.date_to_filename(date)
    #     assert os.path.exists(os.path.join(dataset.dir, filename)), f"Missing {filename}"
    #     temp = xr.open_dataset(os.path.join(dataset.dir, filename), engine='cfgrib')['t2m'].values
    #     np.save(os.path.join(dataset.dir, filename.replace('.grib2', '.npy')), temp)

    # start_date = datetime.datetime(2010, 1, 1, 0, 0)
    # end_date = datetime.datetime(2020, 1, 1, 0, 0) - 4 * datetime.timedelta(hours=3)
    # date_range = pd.date_range(start=start_date, end=end_date, freq='3H')
