from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from horology import Timing

from physics import State

Trajectory: TypeAlias = list[State]


def plot_states(trajectories: list[Trajectory], title: str, filename: str | None = None):
    plt.figure()
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow']
    for i, tr in enumerate(trajectories):
        sns.lineplot(
            x=np.asarray([s.s_x for s in tr]),
            y=np.asarray([s.s_y for s in tr]),
            color=colors[i % len(colors)],
            marker='o'
        )

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

    if filename is not None:
        with Timing(f'Plot saved to {filename} in '):
            plt.savefig(filename)

    plt.show()


def main():
    st_init = State(s_x=0, s_y=0, v_x=10, v_y=10)
    tr1 = [st_init.evolve(t) for t in np.linspace(0, 2 * st_init.half_time(), 50)]
    st_init = State(s_x=0, s_y=0, v_x=8, v_y=12)
    tr2 = [st_init.evolve(t) for t in np.linspace(0, 2 * st_init.half_time(), 100)]
    plot_states([tr1, tr2], 'Trajectories', filename='test.png')


if __name__ == '__main__':
    main()
