import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset

from physics import solution

V0s = [80, 90, 100, 110, 120]
delta_t = 0.02


def create_empirical_data(max_height=50, max_time=10):
    """ Generate data from experiment

    This simulates following situation:
    We lunch a projectile from ground (x=0) with initial speed v0. We measure
    the altitude of the projectile with time resolution of delta_t up to the
    max_height (it's limitation of our measurement device). We repeat this
    experiment for different initial speeds.

    """
    points = []

    for v_0 in V0s:
        t = 0
        x = 0
        while 0 <= x < max_height and t < max_time:
            points.append(((t, v_0), x))
            t += delta_t
            x = solution(t, v_0)
    # points = np.asarray(points, dtype=np.float32)
    print(f'Created empirical data: {len(points)} points')

    return points


def generate_physics_datapoint():
    t = np.random.uniform(0, 10)
    v0 = np.random.uniform(60, 140)
    return t, v0


def create_validation_data():
    points = []
    v_0 = 105
    t = 0
    x = 0
    while 0 <= x:
        points.append(((t, v_0), x))
        t += delta_t
        x = solution(t, v_0)

    # points = np.asarray(points, dtype=np.float32)
    print(f'Created validation data: {len(points)} points')

    return points


class EmpiricalDataset(Dataset):
    def __init__(self):
        self.points = create_empirical_data()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

class PhysicsDataset(IterableDataset):
    def __iter__(self):
        while True:
            yield generate_physics_datapoint()


class ValidationDataset(Dataset):
    def __init__(self):
        self.points = create_validation_data()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return  self.points[idx]


def get_data_loaders():
    train_dataloader = DataLoader(EmpiricalDataset(), batch_size=10, shuffle=True, drop_last=True, num_workers=2)
    physics_dataloader = DataLoader(PhysicsDataset(), batch_size=20, drop_last=True, num_workers=2)
    val_dataloader = DataLoader(ValidationDataset(), batch_size=100)
    return train_dataloader, physics_dataloader, val_dataloader


if __name__ == '__main__':
    print(get_data_loaders())