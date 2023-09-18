from statistics import mean

import numpy as np
import torch
from torch.autograd import grad

from cookbook.rocket_with_linear_air_resistance.vis import plot_dupa
from data import get_data_loaders
from models import TwoSkips


def main(num_epochs=120):
    empirical_dataloader, physics_dataloader, val_dataloader = get_data_loaders()
    model = TwoSkips()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    running_data_loss = None
    running_physics_loss = None

    for epoch in range(num_epochs):
        train_iter = iter(empirical_dataloader)
        physics_iter = iter(physics_dataloader)

        for i in range(len(empirical_dataloader)):
            model.train()

            # train on physics
            optimizer.zero_grad()
            t, v_0 = next(physics_iter)
            t = t.to(torch.float32)
            v_0 = v_0.to(torch.float32)
            t.requires_grad = True
            t.retain_grad()

            output_batch = model(t, v_0)

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

            loss = (
                           (x_grad - xt).pow(2).sum()  # def. of velocity
                           + (xt_grad - xtt).pow(2).sum()  # def. of acceleration
                           + (alpha_grad).pow(2).sum()  # time independence of alpha (i.e. air resistance)
                           + (beta_grad).pow(2).sum()  # time independence of beta (i.e. gravitation)
                           + (xtt + alpha * xt + beta).pow(2).sum()  # differential equation
                   ) * 0.000001

            t.requires_grad = False
            loss.backward()
            optimizer.step()

            if running_physics_loss is None:
                running_physics_loss = loss.item()
            else:
                running_physics_loss = running_physics_loss * 0.99 + loss.item() * 0.01

            # train on data
            optimizer.zero_grad()
            input_batch, target_batch = next(train_iter)

            t, v_0 = input_batch
            t = t.to(torch.float32)
            v_0 = v_0.to(torch.float32)
            target_batch = target_batch.to(torch.float32)

            output_batch = model(t, v_0)

            x = output_batch[:, 0]
            # xt = output_batch[:, 1:2]
            # xtt = output_batch[:, 2:3]
            # alpha = output_batch[:, 3:4]
            # beta = output_batch[:, 4:5]

            loss = torch.nn.functional.mse_loss(x, target_batch)
            loss.backward()
            optimizer.step()

            if running_data_loss is None:
                running_data_loss = loss.item()
            else:
                running_data_loss = running_data_loss * 0.99 + loss.item() * 0.01

            # eval
            # if i % 100 == 0:
            if True:
                x_true = []
                x_pred = []
                ts = []
                model.eval()
                val_losses = []
                for input_batch, target_batch in val_dataloader:
                    t, v_0 = input_batch
                    t = t.to(torch.float32)
                    v_0 = v_0.to(torch.float32)
                    target_batch = target_batch.to(torch.float32)
                    output_batch = model(t, v_0)
                    x = output_batch[:, 0]
                    loss = torch.nn.functional.mse_loss(x, target_batch)
                    val_losses.append(loss.item())
                    x_true.append(target_batch.detach().numpy().squeeze())
                    x_pred.append(x.detach().numpy().squeeze())
                    ts.append(t.detach().numpy().squeeze())
                print(
                    f"Epoch {epoch + 1} / {num_epochs},  Batch {i + 1} / {len(empirical_dataloader)},  "
                    f"edLoss: {running_data_loss:.3g},  "
                    f"phLoss: {running_physics_loss:.3g},  "
                    f"val loss: {mean(val_losses):.3g}"
                )

        if (epoch + 1) % 10 == 0:
            x_true = np.concatenate(x_true)
            y_pred = np.concatenate(x_pred)
            ts = np.concatenate(ts)

            plot_dupa(
                {'x_true': (ts, x_true), 'x_pred': (ts, y_pred)},
                f'Epoch {epoch + 1} / {num_epochs}  Batch {i + 1} / {len(empirical_dataloader)} ',
                # f'data/plot_{epoch + 1}_{i}.png',
            )


if __name__ == '__main__':
    main()
