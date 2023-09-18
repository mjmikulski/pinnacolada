from math import exp


def solution(t: float, v_0: float, alpha: float = 1, beta: float = 10) -> float:
    A = beta + v_0 * alpha
    return - beta / alpha * t + A * (1 - exp(-alpha * t))
