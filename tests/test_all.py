# ruff: noqa: S101
import torch

from rl_utils import calc_return

TOL = 1e-6


def test_calc_return() -> None:
    # Rewards 1, 2, 3 with discount factor 0.9:
    # G = 1 + 2*0.9 + 3*0.9**2
    expected = 1 + 2 * 0.9 + 3 * 0.9**2
    assert (
        abs(calc_return(torch.tensor([1.0, 2.0, 3.0]), 0.9) - expected) < TOL
    )
