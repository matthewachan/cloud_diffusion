from types import SimpleNamespace

import torch
import wandb
from dataset import TemperatureDataset
from torch.utils.data import Subset

from cloud_diffusion.ddpm import ddim_sampler, noisify_ddpm
from cloud_diffusion.models import UNet2D, get_unet_params
from cloud_diffusion.utils import MiniTrainer, NoisifyDataloader, parse_args, set_seed

PROJECT_NAME = "ddpm_clouds"

config = SimpleNamespace(
    epochs=50,  # number of epochs
    dir="/fs/nexus-scratch/mattchan/datasets/hrrr",  # directory of dataset
    model_name="unet_small",  # model name to save [unet_small, unet_big]
    strategy="ddpm",  # strategy to use ddpm
    noise_steps=1000,  # number of noise steps on the diffusion process
    sampler_steps=333,  # number of sampler steps on the diffusion process
    seed=42,  # random seed
    batch_size=128,  # batch size
    img_size=64,  # image size
    device="cuda",  # device
    num_workers=8,  # number of workers for dataloader
    num_frames=4,  # number of frames to use as input
    lr=5e-4,  # learning rate
    validation_days=3,  # number of days to use for validation
    log_every_epoch=5,  # log every n epochs to wandb
    n_preds=8,  # number of predictions to make
)


def train_func(config):
    config.model_params = get_unet_params(config.model_name, config.num_frames)

    set_seed(config.seed)
    device = torch.device(config.device)

    ds = TemperatureDataset(config.dir, config.img_size)
    train_ds = Subset(ds, range(int(len(ds) * 0.8)))
    valid_ds = Subset(ds, range(int(len(ds) * 0.8), len(ds)))

    # DDPM dataloaders
    train_dataloader = NoisifyDataloader(
        train_ds,
        config.batch_size,
        shuffle=True,
        noise_func=noisify_ddpm,
        num_workers=config.num_workers,
    )
    valid_dataloader = NoisifyDataloader(
        valid_ds,
        config.batch_size,
        shuffle=False,
        noise_func=noisify_ddpm,
        num_workers=config.num_workers,
    )

    # model setup
    model = UNet2D(**config.model_params)

    # sampler
    sampler = ddim_sampler(steps=config.sampler_steps)

    # A simple training loop
    trainer = MiniTrainer(train_dataloader, valid_dataloader, model, sampler, device)
    trainer.fit(config)


if __name__ == "__main__":
    parse_args(config)
    with wandb.init(
        project=PROJECT_NAME, config=config, tags=["train", config.model_name]
    ):
        train_func(config)
