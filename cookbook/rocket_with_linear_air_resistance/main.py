import numpy as np
import torch
import wandb
from tqdm import trange

from cookbook.rocket_with_linear_air_resistance.physics import get_physics_loss
from data import get_data_loaders
from models import TwoSkips


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
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0, nesterov=False)

    for epoch in trange(num_epochs):
        for i, (train_batch, physics_batch) in enumerate(zip(empirical_dataloader, physics_dataloader)):
            model.train()

            # train on physics
            optimizer.zero_grad()
            t, v_0 = physics_batch
            t, v_0 = t.to(torch.float32), v_0.to(torch.float32)
            t.requires_grad = True

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
