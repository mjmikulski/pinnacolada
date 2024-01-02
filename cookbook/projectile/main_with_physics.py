from statistics import mean

import torch
from torch.autograd import grad

from data import get_data_loaders
from models import OneSkip
from physics import State
from plotting import plot_states


def main():
    train_dataloader, physics_dataloader, val_dataloader = get_data_loaders(dry_run=False)
    model = OneSkip()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    num_epochs = 5
    running_data_loss = None
    running_physics_loss = None

    for epoch in range(num_epochs):
        train_iter = iter(train_dataloader)
        physics_iter = iter(physics_dataloader)

        for i in range(len(train_dataloader)):
            model.train()

            # train on physics
            optimizer.zero_grad()
            input_batch = next(physics_iter)
            x = input_batch[:, 0:4]
            t = input_batch[:, 4:5]
            t.requires_grad = True
            t.retain_grad()

            output_batch = model(x, t)

            s_x = output_batch[:, 0:1]
            s_y = output_batch[:, 1:2]
            v_x = output_batch[:, 2:3]
            v_y = output_batch[:, 3:4]

            s_x_grad = grad(s_x, t, grad_outputs=torch.ones_like(s_x), create_graph=True)[0]
            s_y_grad = grad(s_y, t, grad_outputs=torch.ones_like(s_y), create_graph=True)[0]
            loss = ((s_x_grad - v_x).pow(2).sum() + (s_y_grad - v_y).pow(2).sum()) * 0.001

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
            x = input_batch[:, 0:4]
            t = input_batch[:, 4:5]
            output_batch = model(x, t)
            loss = torch.nn.functional.mse_loss(output_batch, target_batch)
            loss.backward()
            optimizer.step()

            if running_data_loss is None:
                running_data_loss = loss.item()
            else:
                running_data_loss = running_data_loss * 0.99 + loss.item() * 0.01

            # eval
            if i % 100 == 0:
                model.eval()
                val_losses = []
                for input_batch, target_batch in val_dataloader:
                    x = input_batch[:, 0:4]
                    t = input_batch[:, 4:5]
                    output_batch = model(x, t)
                    loss = torch.nn.functional.mse_loss(output_batch, target_batch)
                    val_losses.append(loss.item())
                print(
                    f"Epoch {epoch + 1} / {num_epochs},  Batch {i} / {len(train_dataloader)},  "
                    f"dLoss: {running_data_loss:.3g},  "
                    f"phLoss: {running_physics_loss:.3g},  "
                    f"val loss: {mean(val_losses):.3g}"
                )

        print('\n')

        model.eval()
        input, target = next(iter(val_dataloader))
        x = input[:, 0:4]
        t = input[:, 4:5]
        output = model(x, t)
        plot_states(
            [
                [State(*q.detach().numpy()[:4]) for q in target],
                [State(*q.detach().numpy()[:4]) for q in output]
            ],
            f'Epoch {epoch + 1} / {num_epochs}',
            f'data/epoch_{epoch + 1}_of_{num_epochs}.png'
        )


if __name__ == '__main__':
    main()
