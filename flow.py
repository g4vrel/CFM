import torch
from torch import nn, Tensor
from torchdiffeq import odeint


class OptimalTransportFlow:
    def __init__(self, sigma_min: float = 1e-2):
        super().__init__()
        self.sigma_min = sigma_min

    @torch.compile
    def step(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        t = t[:, None, None, None]
        mu = t * x1
        sigma = 1 - (1 - self.sigma_min) * t
        return sigma * x0 + mu

    @torch.compile
    def target(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        return x1 - (1 - self.sigma_min) * x0


@torch.inference_mode()
def sample_images(
    model: nn.Module, shape: tuple = (64, 3, 32, 32), num_steps: int = 5, device="cuda"
):
    model.eval()

    x0 = torch.randn(shape, device=device)
    timesteps = torch.linspace(0.0, 1.0, num_steps, device=device)

    samples = odeint(
        func=lambda t, x: model(x, t.repeat(shape[0])),
        t=timesteps,
        y0=x0,
        method="dopri5",
        atol=1e-5,
        rtol=1e-5,
    )
    return samples
