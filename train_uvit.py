from types import SimpleNamespace

import torch
import wandb
from dataset import TemperatureDataset
from torch.utils.data import Subset

from cloud_diffusion.models import UViT, get_uvit_params
from cloud_diffusion.simple_diffusion import noisify_uvit, simple_diffusion_sampler
from cloud_diffusion.utils import MiniTrainer, NoisifyDataloader, parse_args, set_seed

DEBUG = True
PROJECT_NAME = "ddpm_clouds"

config = SimpleNamespace(
    epochs=100,  # number of epochs
    model_name="uvit_small",  # model name to save
    dir="/fs/nexus-scratch/mattchan/datasets/hrrr-west",  # directory of dataset
    region="west",  # region of dataset [west, east]
    strategy="simple_diffusion",  # strategy to use [ddpm, simple_diffusion]
    noise_steps=1000,  # number of noise steps on the diffusion process
    sampler_steps=500,  # number of sampler steps on the diffusion process
    seed=42,  # random seed
    batch_size=6,  # batch size
    img_size=512,  # image size
    device="cuda",  # device
    num_workers=8,  # number of workers for dataloader
    num_frames=4,  # number of frames to use as input
    lr=5e-4,  # learning rate
    validation_days=3,  # number of days to use for validation
    n_preds=8,  # number of predictions to make
    log_every_epoch=5,  # log every n epochs to wandb
)


def train_func(config):
    config.model_params = get_uvit_params(config.model_name, config.num_frames)
    config.model_params = dict(
        dim=256,
        ff_mult=2,
        vit_depth=4,
        channels=4,
        patch_size=4,
        final_img_itransform=torch.nn.Conv2d(config.num_frames, 1, 1),
    )

    set_seed(config.seed)
    device = torch.device(config.device)

    ds = TemperatureDataset(config.dir, config.img_size)
    train_ds = Subset(ds, range(int(len(ds) * 0.8)))
    valid_ds = Subset(ds, range(int(len(ds) * 0.8), len(ds)))

    # UViT dataloaders
    train_dataloader = NoisifyDataloader(
        train_ds,
        config.batch_size,
        shuffle=True,
        noise_func=noisify_uvit,
        num_workers=config.num_workers,
    )
    valid_dataloader = NoisifyDataloader(
        valid_ds,
        config.batch_size,
        shuffle=False,
        noise_func=noisify_uvit,
        num_workers=config.num_workers,
    )
    # model setup
    model = UViT(**config.model_params)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))

    # sampler
    sampler = simple_diffusion_sampler(steps=config.sampler_steps)

    # A simple training loop
    trainer = MiniTrainer(train_dataloader, valid_dataloader, model, sampler, device)
    trainer.fit(config)


if __name__ == "__main__":
    parse_args(config)
    with wandb.init(
        project=PROJECT_NAME,
        config=config,
        tags=["train", config.region, config.model_name],
    ):
        train_func(config)
