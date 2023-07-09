import glob
import os

import numpy as np
import torch
import torchvision.transforms as T

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


if __name__ == "__main__":
    dataset = TemperatureDataset(64)
    print(dataset[0].shape)
