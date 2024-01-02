from math import cos, pi, sin

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from physics import State


def generate_training_data(dry_run=False):
    step = 10 if dry_run else 1
    points = []
    for alpha_deg in range(10, 81):
        alpha = alpha_deg / 180 * pi

        for v_0 in range(5, 25):
            init_state = State(s_x=0, s_y=0, v_x=v_0 * cos(alpha), v_y=v_0 * sin(alpha))

            for t_perc in range(0, 111, step):
                t = init_state.half_time() * t_perc / 100
                state = init_state.evolve(t)
                points.append(
                    (
                        (*init_state, t), (*state, t)
                    )
                )

    points = np.asarray(points, dtype=np.float32)
    print(f'Generated training data, shape: {points.shape}')
    return points


def generate_physics_data(dry_run=False):
    step = 10 if dry_run else 1
    points = []
    for alpha_deg in range(0, 180, 5):
        alpha = alpha_deg / 180 * pi

        for v_0 in range(1, 50):
            init_state = State(s_x=0, s_y=0, v_x=v_0 * cos(alpha), v_y=v_0 * sin(alpha))

            for t_perc in range(0, 301, step):
                t = init_state.half_time() * t_perc / 100

                points.append((*init_state, t))

    points = np.asarray(points, dtype=np.float32)
    print(f'Generated physics data, shape: {points.shape}')
    return points


def generate_validation_data(dry_run=False):
    step = 10 if dry_run else 1
    points = []
    init_state = State(s_x=0, s_y=0, v_x=10, v_y=10)

    for t_perc in range(0, 201, step):
        t = init_state.half_time() * t_perc / 100
        state = init_state.evolve(t)
        points.append(
            (
                (*init_state, t), (*state, t)
            )
        )

    points = np.asarray(points, dtype=np.float32)
    print(f'Generated validation data, shape: {points.shape}')
    return points


class TrainDataset(Dataset):
    def __init__(self, dry_run):
        super().__init__()
        self.points = generate_training_data(dry_run)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        input, target = self.points[idx]
        return torch.tensor(input), torch.tensor(target)


class PhysicsDataset(Dataset):
    def __init__(self, dry_run):
        super().__init__()
        self.points = generate_physics_data(dry_run)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        input = self.points[idx]
        return torch.tensor(input)


class ValidationDataset(Dataset):
    def __init__(self, dry_run):
        super().__init__()
        self.points = generate_validation_data(dry_run)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        input, target = self.points[idx]
        return torch.tensor(input), torch.tensor(target)


def get_data_loaders(dry_run=False):
    train_dataloader = DataLoader(TrainDataset(dry_run), batch_size=64, shuffle=True, drop_last=True, num_workers=2)
    physics_dataloader = DataLoader(PhysicsDataset(dry_run), batch_size=64, shuffle=True, drop_last=True, num_workers=2)
    val_dataloader = DataLoader(ValidationDataset(dry_run), batch_size=200, shuffle=False)
    return train_dataloader, physics_dataloader, val_dataloader


if __name__ == '__main__':
    print(get_data_loaders(dry_run=True))
