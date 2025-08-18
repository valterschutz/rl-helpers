# pyright: reportUnknownVariableType=none, reportUnknownMemberType=none, reportUnknownArgumentType=none, reportMissingTypeStubs=none

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable, Sequence

import numpy as np
from tensordict import TensorDictBase
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import EnvBase
from torch.distributions import Categorical
import torch.nn.functional as F


def module_norm(module: nn.Module) -> float:
    """
    Compute the L2 norm of all parameters in a module.

    Args:
        module: nn.Module to compute the parameter norm for.

    Returns:
        float: L2 norm of all parameters.

    """
    # Aggregate all parameters from the module and compute the norm
    return torch.norm(
        torch.cat([p.view(-1) for p in module.parameters()]), p=2
    ).item()  # L2 norm (Euclidean norm)


@contextmanager
def eval_mode(*models: nn.Module) -> Iterator[None]:
    """
    Context manager to temporarily set models to evaluation mode.

    Args:
        *models: One or more nn.Module objects.

    Yields:
        None. Restores original training modes on exit.

    """
    was_training = [model.training for model in models]
    try:
        for model in models:
            model.eval()
        yield
    finally:
        for model, mode in zip(models, was_training, strict=False):
            model.train(mode)


def dict_with_prefix(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    """Return a dictionary with all keys prefixed by `prefix`."""
    return {f"{prefix}{k}": v for k, v in d.items()}


# tested
def calc_return(tensor: torch.Tensor, gamma: float) -> float:
    """Calculate the return for the rewards in `tensor` with discount factor `gamma`."""
    n_steps = len(tensor)
    discounting = (gamma * torch.ones(n_steps)) ** torch.arange(n_steps)
    g = (discounting * tensor).sum().item()
    return g


def verify_return(
    env: EnvBase,
    policy: Callable[[TensorDictBase], TensorDictBase],
    gamma: float,
    expected_return: float,
    n_evals: int,
    mean_tolerance: float,
    std_tolerance: float,
    max_steps: int,
) -> bool:
    """Test whether rollouts with `policy` in `env` have the `expected_return`."""
    returns: list[float] = []
    for _ in range(n_evals):
        td = env.rollout(max_steps, policy)
        returns.append(calc_return(td["next", "reward"], gamma))

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))

    return (
        abs(mean_return - expected_return) < mean_tolerance
        and std_return < std_tolerance
    )


# def multinomial_masks(
#     batch_size: torch.Size,
#     probs: Sequence[float],
#     device: torch.device | None = None,
#     generator: torch.Generator | None = None,
# ) -> tuple[torch.Tensor, ...]:
#     """
#     Sample mutually exclusive boolean masks according to given probabilities.
#
#     Args:
#         batch_size: torch.Size, e.g. (N,) or (N, M) — leading dimensions for sampling.
#         probs: sequence of probabilities that should sum to 1.
#         device: optional torch device.
#         generator: optional torch.Generator for reproducibility.
#
#     Returns:
#         Tuple of boolean masks, one per category, shape = [*batch_size].
#     """
#     dist = Categorical(probs=torch.tensor(probs, device=device))
#     samples = dist.sample(batch_size, generator=generator)  # integer categories
#     masks = F.one_hot(samples, num_classes=len(probs)).to(torch.bool)
#
#     return tuple(masks[..., i] for i in range(len(probs)))$


def multinomial_masks(
    batch_size: torch.Size,
    probs: Sequence[float],
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    Sample mutually exclusive boolean masks according to given probabilities.

    Args:
        batch_size: torch.Size, e.g. (N,) or (N, M).
        probs: sequence of probabilities that should sum to 1.
        device: optional torch device.
        generator: optional torch.Generator for reproducibility.

    Returns:
        Tuple of boolean masks, one per category, shape = [*batch_size].
    """
    probs_t = torch.tensor(probs, device=device)
    # Flatten batch for multinomial sampling
    numel = int(torch.tensor(batch_size).prod())
    samples = torch.multinomial(
        probs_t, num_samples=numel, replacement=True, generator=generator
    )
    samples = samples.view(*batch_size)  # reshape back to batch

    masks = F.one_hot(samples, num_classes=len(probs)).to(torch.bool)
    return tuple(masks[..., i] for i in range(len(probs)))
