import math

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from einops import rearrange
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets

unloader = v2.Compose(
    [
        v2.Lambda(lambda t: (t + 1) * 0.5),
        v2.Lambda(lambda t: t.permute(0, 2, 3, 1)),
        v2.Lambda(lambda t: t * 255.0),
    ]
)


def make_im_grid(x0: torch.Tensor, xy: tuple = (1, 10)):
    x, y = xy
    im = unloader(x0.cpu())
    B, C, H, W = x0.shape
    im = (
        rearrange(im, "(x y) h w c -> (x h) (y w) c", x=B // x, y=B // y)
        .numpy()
        .astype(np.uint8)
    )
    im = v2.ToPILImage()(im)
    return im


def get_loaders(config, root):
    size = config.data.img_size
    bs = config.data.batch_size
    nw = config.data.num_workers
    name = config.data.dataset.lower()

    base_train_tf = [
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    base_test_tf = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]

    if name == "stl10":
        resize_tf = [v2.Resize(size, antialias=True), v2.CenterCrop(size)]
        train_tf = v2.Compose(resize_tf + base_train_tf)
        test_tf = v2.Compose(resize_tf + base_test_tf)
        train_set = datasets.STL10(
            root, split="unlabeled", download=True, transform=train_tf
        )
        test_set = datasets.STL10(root, split="test", download=True, transform=test_tf)

    elif name == "cifar10":
        train_tf = v2.Compose(base_train_tf)
        test_tf = v2.Compose(base_test_tf)
        train_set = datasets.CIFAR10(
            root, train=True, download=True, transform=train_tf
        )
        test_set = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    else:
        raise ValueError(
            f"Unknown dataset '{config.data.dataset}'. Use 'cifar10' or 'stl10'."
        )

    train_loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
        prefetch_factor=4,
    )

    return train_loader, test_loader


def make_checkpoint(path, step, epoch, model, optim=None, scaler=None, ema_model=None):
    checkpoint = {
        "epoch": int(epoch),
        "step": int(step),
        "model_state_dict": model.state_dict(),
    }

    if optim is not None:
        checkpoint["optim_state_dict"] = optim.state_dict()

    if ema_model is not None:
        checkpoint["ema_model_state_dict"] = ema_model.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optim=None, scaler=None, ema_model=None):
    checkpoint = torch.load(path, weights_only=True)
    step = int(checkpoint["step"])
    epoch = int(checkpoint["epoch"])

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if optim is not None:
        optim.load_state_dict(checkpoint["optim_state_dict"])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        ema_model.eval()

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    model.eval()

    return step, epoch, model, optim, scaler, ema_model


def print_steps_info(cfg: DictConfig, loader: DataLoader):
    batches_per_epoch = len(loader)
    effective_samples = batches_per_epoch * loader.batch_size
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / cfg.trainer.accumulation_steps
    )

    print(
        f"samples/epoch = {effective_samples:,}  |  "
        f"batches/epoch = {batches_per_epoch:,}  |  "
        f"optimizer-steps/epoch = {optimizer_steps_per_epoch:,}  "
        f"(accum_steps = {cfg.trainer.accumulation_steps})"
    )

    return effective_samples, batches_per_epoch, optimizer_steps_per_epoch
