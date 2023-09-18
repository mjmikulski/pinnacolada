import numpy as np
import torch
import wandb
from torch.autograd import grad

from data import get_data_loaders
from models import TwoSkips

PHYSICS_LOSS_WEIGHT = 0.01


def get_physics_loss(t, output_batch):
    x = output_batch[:, 0:1]
    xt = output_batch[:, 1:2]
    xtt = output_batch[:, 2:3]
    alpha = output_batch[:, 3:4]
    beta = output_batch[:, 4:5]

    x_sum = x.sum()
    x_grad = grad(x_sum, t, create_graph=True)[0]

    xt_sum = xt.sum()
    xt_grad = grad(xt_sum, t, create_graph=True)[0]

    alpha_sum = alpha.sum()
    alpha_grad = grad(alpha_sum, t, create_graph=True)[0]

    beta_sum = beta.sum()
    beta_grad = grad(beta_sum, t, create_graph=True)[0]

    def_velocity = (x_grad - xt).pow(2).mean()
    def_acceleration = (xt_grad - xtt).pow(2).mean()
    time_independence_of_alpha = (alpha_grad).pow(2).mean()
    time_independence_of_beta = (beta_grad).pow(2).mean()
    diff_equation = torch.log((xtt + alpha * xt + beta).pow(2)).mean()

    loss = PHYSICS_LOSS_WEIGHT * (
            def_velocity
            + def_acceleration
            + time_independence_of_alpha
            + time_independence_of_beta
            + diff_equation
    )

    wandb.log({
        'defs': {
            'velocity': def_velocity.item(),
            'acceleration': def_acceleration.item(),
        },

        'alpha': {
            'min': alpha.min().item(),
            'mean': alpha.mean().item(),
            'max': alpha.max().item(),
            't_int': time_independence_of_alpha.item(),
        },
        'beta': {
            'min': beta.min().item(),
            'mean': beta.mean().item(),
            'max': beta.max().item(),
            't_int': time_independence_of_beta.item(),
        },

        'diff_equation': diff_equation.item(),
        'total_physics_loss': loss.item(),
    }, commit=False)

    return loss


def eval(epoch, num_epochs, model, val_dataloader):
    model.eval()
    x_true = []
    x_pred = []
    ts = []
    val_losses = []
    for input_batch, target_batch in val_dataloader:
        t, v_0 = input_batch
        t, v_0, target_batch = t.to(torch.float32), v_0.to(torch.float32), target_batch.to(torch.float32)

        output_batch = model(t, v_0)
        x = output_batch[:, 0]
        loss = torch.nn.functional.mse_loss(x, target_batch)

        val_losses.append(loss.item())
        x_true.append(target_batch.detach().numpy().squeeze())
        x_pred.append(x.detach().numpy().squeeze())
        ts.append(t.detach().numpy().squeeze())
    x_true = np.concatenate(x_true)
    x_pred = np.concatenate(x_pred)
    ts = np.concatenate(ts)
    val_loss = np.mean(val_losses)
    wandb.log({'val_loss': val_loss,
               'main_plot': wandb.plot.line_series(
                   xs=ts,
                   ys=[x_true, x_pred],
                   keys=['x_true', 'x_pred'],
                   title=f'Epoch {epoch + 1} / {num_epochs}',
                   xname='time',
               )
               })


def main(num_epochs=120):
    empirical_dataloader, physics_dataloader, val_dataloader = get_data_loaders()
    model = TwoSkips()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(num_epochs):
        for i, (train_batch, physics_batch) in enumerate(zip(empirical_dataloader, physics_dataloader)):
            model.train()

            # train on physics
            optimizer.zero_grad()
            t, v_0 = physics_batch
            t, v_0 = t.to(torch.float32), v_0.to(torch.float32)
            t.requires_grad = True
            t.retain_grad()

            output_batch = model(t, v_0)
            loss = get_physics_loss(t, output_batch)

            t.requires_grad = False
            loss.backward()
            optimizer.step()

            # train on data
            optimizer.zero_grad()
            input_batch, target_batch = train_batch
            t, v_0 = input_batch
            t, v_0, target_batch = t.to(torch.float32), v_0.to(torch.float32), target_batch.to(torch.float32)

            x = model(t, v_0)[:, 0]

            loss = torch.nn.functional.mse_loss(x, target_batch)
            loss.backward()
            optimizer.step()

            wandb.log({'data_loss': loss.item()})

        eval(epoch, num_epochs, model, val_dataloader)


if __name__ == '__main__':
    # wandb.init(project='pinnacolada', mode='disabled')
    wandb.init(project='pinnacolada')
    main()
