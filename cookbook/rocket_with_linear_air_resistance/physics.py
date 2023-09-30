from math import exp

import torch
import wandb
from torch.autograd import grad

PHYSICS_LOSS_WEIGHT = 0.01


def solution(t: float, v_0: float, alpha: float = 1, beta: float = 10) -> float:
    A = beta + v_0 * alpha
    return - beta / alpha * t + A * (1 - exp(-alpha * t))


v_ch = 100
t_ch = 5

velocity_rescaler = 1 / v_ch
acceleration_rescaler = t_ch / v_ch
alpha_rescaler = t_ch ** 2
beta_rescaler = t_ch ** 2 / v_ch
ode_rescaler = acceleration_rescaler


def get_physics_loss(t: torch.Tensor, output_batch: torch.Tensor) -> torch.Tensor:
    x, xt, xtt, alpha, beta = (output_batch[:, i] for i in range(5))

    x_sum = x.sum()  # maybe is_grads_batched instead of sum?
    x_grad = grad(x_sum, t, create_graph=True)[0]

    xt_sum = xt.sum()
    xt_grad = grad(xt_sum, t, create_graph=True)[0]

    alpha_sum = alpha.sum()
    alpha_grad = grad(alpha_sum, t, create_graph=True)[0]

    beta_sum = beta.sum()
    beta_grad = grad(beta_sum, t, create_graph=True)[0]

    loss_velocity = ((x_grad - xt) * velocity_rescaler).pow(2).mean()
    loss_acceleration = ((xt_grad - xtt) * acceleration_rescaler).pow(2).mean()
    loss_alpha = (alpha_grad * alpha_rescaler).pow(2).mean()
    loss_beta = (beta_grad * beta_rescaler).pow(2).mean()
    loss_ode = ((xtt + alpha * xt + beta) * ode_rescaler).pow(2).mean()

    loss = PHYSICS_LOSS_WEIGHT * (
            loss_velocity
            + loss_acceleration
            + loss_alpha
            + loss_beta
            + loss_ode
    )

    wandb.log({
        'loss': {
            'velocity': loss_velocity.item(),
            'acceleration': loss_acceleration.item(),
            'alpha': loss_alpha.item(),
            'beta': loss_beta.item(),
            'ode': loss_ode.item(),
        },

        'alpha': {
            'min': alpha.min().item(),
            'mean': alpha.mean().item(),
            'max': alpha.max().item(),
        },
        'beta': {
            'min': beta.min().item(),
            'mean': beta.mean().item(),
            'max': beta.max().item(),
        },
    }, commit=False)

    return loss
