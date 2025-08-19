import torch
from torch import nn
from torchrl.modules import MLP

from rl_helpers import calc_return, module_norm

TOL = 1e-6


def test_calc_return() -> None:
    # Rewards 1, 2, 3 with discount factor 0.9:
    # G = 1 + 2*0.9 + 3*0.9**2
    expected = 1 + 2 * 0.9 + 3 * 0.9**2
    assert (
        abs(calc_return(torch.tensor([1.0, 2.0, 3.0]), 0.9) - expected) < TOL
    )


def test_module_norm() -> None:
    # We merely test that module_norm returns a number (and does not crash) for both nn.Linear and torchrl.modules.MLP
    module1 = nn.Linear(3, 4)
    norm1 = module_norm(module1)
    print(f"{norm1=}")

    module2 = MLP(in_features=3, out_features=4, num_cells=[2, 2])
    norm2 = module_norm(module2)
    print(f"{norm2=}")
