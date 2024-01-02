from __future__ import annotations

from typing import NamedTuple

G = 10  # m/s^2


class State(NamedTuple):
    s_x: float
    s_y: float
    v_x: float
    v_y: float

    def evolve(self, t: float) -> State:
        return State(
            s_x=self.s_x + self.v_x * t,
            s_y=self.s_y + self.v_y * t - 1 / 2 * G * t ** 2,
            v_x=self.v_x,
            v_y=self.v_y - G * t
        )

    def half_time(self) -> float:
        return self.v_y / G
