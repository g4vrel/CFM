import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch import Tensor
from flow import OptimalTransportFlow, sample_images
from unet import Unet
from utils import *


def _data_root_and_download(cfg):
    root = Path(get_original_cwd()) / cfg.data.root
    root.mkdir(parents=True, exist_ok=True)
    download = bool(cfg.data.download) and not any(root.iterdir())
    cfg.data.root = str(root)
    cfg.data.download = download
    return cfg


def set_flags():
    """Set performance flags and seed."""
    torch.manual_seed(159753)
    np.random.seed(159753)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def get_loss_fn(model: Unet, flow: OptimalTransportFlow):
    mse = torch.nn.MSELoss()

    def loss_fn(batch: Tensor) -> Tensor:
        t = torch.rand(batch.shape[0], device=batch.device)
        x0 = torch.randn_like(batch)

        xt = flow.step(t, x0, batch)
        pred_vel = model(xt, t)
        true_vel = flow.target(t, x0, batch)

        loss = mse(pred_vel, true_vel)
        return loss

    return loss_fn


def get_lr(cfg: DictConfig, step: int) -> float:
    """
    Linear warm-up followed by linear decay back to min_lr until max_steps.
    Caps at min_lr after that.
    """
    min_lr, max_lr = cfg.trainer.min_lr, cfg.trainer.max_lr
    warmup, max_steps = cfg.trainer.warmup_steps, cfg.trainer.max_steps

    if step < warmup:
        lr = min_lr + (max_lr - min_lr) * step / warmup
    elif step <= max_steps:
        decay_ratio = (step - warmup) / (max_steps - warmup)
        lr = max_lr - (max_lr - min_lr) * decay_ratio
    else:
        lr = min_lr

    return max(min_lr, min(lr, max_lr))


@torch.no_grad()
def eval_sample(cfg: DictConfig, epoch: int, model, ema_model) -> None:
    model.eval()
    ema_model.eval()

    print(f"Generating samples at epoch {epoch}")
    shape = (64, 3, cfg.sample.size, cfg.sample.size)

    gen_x = sample_images(model, shape, num_steps=2)
    gen_x_ema = sample_images(ema_model, shape, num_steps=2)
    gen_x = gen_x[-1]
    gen_x_ema = gen_x_ema[-1]

    assert gen_x.shape == shape

    image = make_im_grid(gen_x, (8, 8))
    image.save(f"samples/{epoch}.png")
    image_ema = make_im_grid(gen_x_ema, (8, 8))
    image_ema.save(f"samples/ema_{epoch}.png")


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    os.makedirs("samples", exist_ok=True)

    set_flags()
    cfg = _data_root_and_download(cfg)
    device = "cuda"

    model = Unet(ch=128, att_channels=[0, 1, 1, 0], dropout=0.0).to(device)
    model = torch.compile(model)
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
    )
    flow = OptimalTransportFlow(cfg.flow.sigma_min)

    loss_fn = get_loss_fn(model, flow)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.trainer.min_lr)
    train_loader, _ = get_loaders(cfg, cfg.data.root)
    scaler = torch.amp.GradScaler()

    print_steps_info(cfg, train_loader)

    ckpt = None
    if ckpt is not None:
        step, curr_epoch, model, optim, scaler, ema_model = load_checkpoint(
            ckpt, model, optim, scaler, ema_model
        )
        print(f"Loaded checkpoint [step {step} ({curr_epoch})]")
    else:
        step = 0
        curr_epoch = 0

    accumulation_steps = int(cfg.trainer.accumulation_steps)

    for epoch in range(curr_epoch, cfg.trainer.epochs + 1):
        model.train()
        ema_model.train()

        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)

            if i % accumulation_steps == 0:
                optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device):
                loss = loss_fn(x) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optim)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optim)
                scaler.update()

                ema_model.update_parameters(model)

                for g in optim.param_groups:
                    lr = get_lr(cfg, step)
                    g["lr"] = lr

                if (step + 1) % cfg.trainer.log_freq == 0:
                    true_loss = loss.item() * accumulation_steps
                    print(
                        f"Step: {step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad.item():.5f} | Lr: {lr:.3e}"
                    )

                step += 1

        eval_sample(cfg, epoch, model, ema_model)

    make_checkpoint(f"ckp_{step}.tar", step, epoch, model, optim, scaler, ema_model)


if __name__ == "__main__":
    main()
