import io
import logging
import random
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import wandb
from fastprogress import progress_bar
from herbie import FastHerbie
from paint.standard2 import cm_tmp
from toolbox import EasyMap, pc

from cloud_diffusion.ddpm import ddim_sampler
from cloud_diffusion.models import UNet2D, get_unet_params
from cloud_diffusion.utils import MAX_TEMP, MIN_TEMP, parse_args, rescale, set_seed
from cloud_diffusion.wandb import vhtile

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROJECT_NAME = "ddpm_clouds"
JOB_TYPE = "inference"
# East coast model
MODEL_ARTIFACT = "matthewachan/ddpm_clouds/69edc463_unet_small:v0"  # small model
# West coast model
MODEL_ARTIFACT = "matthewachan/ddpm_clouds/selw4oaq_unet_small:v0"  # small model

config = SimpleNamespace(
    model_name="unet_small",  # model name to save [unet_small, unet_big]
    region="west",  # east or west
    sampler_steps=333,  # number of sampler steps on the diffusion process
    num_frames=4,  # number of frames to use as input,
    img_size=256,  # image size to use
    num_random_experiments=12,  # we will perform inference multiple times on the same inputs
    seed=33,
    device="cuda" if torch.cuda.is_available() else "cpu",
    sampler="ddim",
    future_frames=4,  # number of future frames
    bs=1,  # how many samples
)


def scale(arr):
    "Scales values of array in [0,1]"
    m, M = arr.min(), arr.max()
    return (arr - m) / (M - m)


class Inference:
    def __init__(self, config):
        self.config = config
        set_seed(config.seed)

        # create a batch of data to use for inference
        self.prepare_data()

        # we default to ddim as it's faster and as good as ddpm
        self.sampler = ddim_sampler(config.sampler_steps)

        # create the Unet
        model_params = get_unet_params(config.model_name, config.num_frames)

        logger.info(
            f"Loading model {config.model_name} from artifact: {MODEL_ARTIFACT}"
        )
        self.model = UNet2D.from_artifact(model_params, MODEL_ARTIFACT).to(
            config.device
        )

        self.loss_func = torch.nn.MSELoss(reduction="none")

        self.model.eval()

    def prepare_data(self):
        "Generates a batch of data from the validation dataset"

        # Download HRRR data for inference
        start = pd.to_datetime("2022-12-21")
        end = pd.to_datetime("2022-12-25")
        three_days = pd.date_range(start=start, end=end, freq="4H")
        dl_dir = "/fs/nexus-scratch/mattchan/datasets/hrrr"
        ds = FastHerbie(
            three_days, model="hrrr", product="sfc", save_dir=dl_dir
        ).xarray("TMP:2 m")

        # Crop the data to the region of interest
        tfm = T.Resize((256, 256), antialias=True)
        if self.config.region == "east":
            data = ds.t2m[:, :1024, -1024:].to_numpy()
            fmt = lambda x: tfm(
                torch.from_numpy(x.to_numpy()[:1024, -1024:]).unsqueeze(0)
            ).squeeze(0)
        elif self.config.region == "west":
            data = ds.t2m[:, :1024, :1024].to_numpy()
            fmt = lambda x: tfm(
                torch.from_numpy(x.to_numpy()[:1024, :1024]).unsqueeze(0)
            ).squeeze(0)

        # Split the data into 4 frame sequences
        wds = np.lib.stride_tricks.sliding_window_view(data, 4, axis=0).transpose(
            (0, 3, 1, 2)
        )
        wds = 0.5 - rescale(wds, MIN_TEMP, MAX_TEMP)
        wds = tfm(torch.from_numpy(wds))

        # Preprocess and extract data
        self.dates = ds.time.values
        self.proj = ds.herbie.crs
        self.lat = fmt(ds.latitude)
        self.long = fmt(ds.longitude)
        self.valid_ds = wds
        self.idxs = random.choices(
            range(len(self.valid_ds) - 4 - config.future_frames), k=config.bs
        )  # select some samples
        self.batch = self.valid_ds[self.idxs].to(config.device)
        print(self.idxs)

    def sample_more(self, frames, future_frames=1):
        "Autoregressive sampling, starting from `frames`. It is hardcoded to work with 3 frame inputs."
        for _ in progress_bar(range(future_frames), total=future_frames, leave=True):
            # compute new frame with previous 3 frames
            new_frame = self.sampler(self.model, frames[:, -3:, ...])
            # add new frame to the sequence
            frames = torch.cat([frames, new_frame.to(frames.device)], dim=1)
        return frames.cpu()

    def forecast(self):
        "Perform inference on the batch of data."
        logger.info(
            f"Forecasting {self.batch.shape[0]} samples for {self.config.future_frames} future frames."
        )
        sequences = []
        for i in range(self.config.num_random_experiments):
            logger.info(
                f"Generating {i+1}/{self.config.num_random_experiments} futures."
            )
            frames = self.sample_more(self.batch, self.config.future_frames)
            sequences.append(frames)

        return sequences

    def log_to_wandb(self, sequences):
        "Create a table with the ground truth and the generated frames. Log it to wandb."
        table = wandb.Table(
            columns=[
                "date",
                *[f"mse_{i}" for i in range(config.num_random_experiments)],
                "gt",
                *[f"gen_{i}" for i in range(config.num_random_experiments)],
                "gt/gen",
            ]
        )

        def make_imgs(data):
            imgs = np.array(
                [visualize(frame, self.lat, self.long, self.proj) for frame in data]
            ).astype(float)
            return imgs

        def make_vid(data):
            imgs = make_imgs(data).astype(np.uint8)
            return wandb.Video(imgs)

        for i, idx in enumerate(self.idxs):
            gt_vid = make_vid(
                self.valid_ds[idx : idx + 4 + config.future_frames, 0, ...]
            )
            pred_vids = [make_vid(frames[i]) for frames in sequences]
            gt_imgs = torch.from_numpy(
                make_imgs(self.valid_ds[idx : idx + 4 + config.future_frames, 0, ...])
            ).long()
            pred_imgs = [torch.from_numpy(make_imgs(frames[i])) for frames in sequences]
            loss = [
                np.array2string(
                    torch.sum(self.loss_func(gt_imgs, pred_img), dim=(1, 2, 3))
                    .detach()
                    .cpu()
                    .numpy()[4:],
                    precision=2,
                    separator=",",
                    suppress_small=True,
                )
                for pred_img in pred_imgs
            ]
            gt_gen = wandb.Image(vhtile(gt_imgs, *pred_imgs))
            table.add_data(self.dates[idx], *loss, gt_vid, *pred_vids, gt_gen)
        logger.info("Logging results to wandb...")
        wandb.log({f"gen_table_{config.future_frames}_random": table})


def visualize(temp, lat, long, proj):
    def k_to_f(k):
        return (k - 273.15) * (9 / 5) + 32

    # Renormalize temperatuer values to Kelvin
    temp = (0.5 - temp) * (MAX_TEMP - MIN_TEMP) + MIN_TEMP
    # Convert Fahrenheit
    temp = k_to_f(temp)

    # Create plot of the US
    fig = plt.figure()
    ax = plt.axes(projection=proj)
    ax = EasyMap("50m", ax=ax, crs=proj, figsize=[10, 8]).STATES(alpha=0.9).ax
    color = cm_tmp(units="F")

    # Add temperature data
    mesh = ax.pcolormesh(long, lat, temp, transform=pc, **color.cmap_kwargs)
    # plt.colorbar(mesh, orientation="horizontal", pad=0.01, **color.cbar_kwargs)

    # Convert plot to numpy array
    io_buf = io.BytesIO()
    b = ax.get_window_extent()
    fig.savefig(io_buf, format="raw", bbox_inches="tight", pad_inches=0)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(b.height), int(b.width), -1),
    )
    plt.close()
    io_buf.close()
    return img_arr.transpose(2, 0, 1)


if __name__ == "__main__":

    parse_args(config)
    set_seed(config.seed)

    with wandb.init(
        project=PROJECT_NAME,
        job_type=JOB_TYPE,
        config=config,
        tags=["test", config.model_name],
    ):
        infer = Inference(config)
        sequences = infer.forecast()
        infer.log_to_wandb(sequences)
