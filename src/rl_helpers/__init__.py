# pyright: reportUnknownVariableType=none, reportUnknownMemberType=none, reportUnknownArgumentType=none, reportMissingTypeStubs=none

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torch import nn
from torchrl.envs import EnvBase


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


T = TypeVar("T")


def dict_with_prefix(prefix: str, d: dict[str, T]) -> dict[str, T]:
    """Return a dictionary with all keys prefixed by `prefix`."""
    return {f"{prefix}{k}": v for k, v in d.items()}


# tested
def calc_return(tensor: torch.Tensor, gamma: float) -> float:
    """Calculate the return for the rewards in `tensor` with discount factor `gamma`."""
    n_steps = len(tensor)
    discounting = (
        gamma * torch.ones(n_steps, device=tensor.device)
    ) ** torch.arange(n_steps, device=tensor.device)
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


# tested
def calc_rollout_returns(td: TensorDictBase, gamma: float) -> torch.Tensor:
    """Calculate the return for each rollout trajectory in `td`."""
    assert len(td.batch_size) == 2, "Only 2D batch size supported"
    n_trajectories = td.batch_size[0]

    returns = torch.empty(n_trajectories)
    # Trajectories end at different time steps. Find the terminating time step for each trajectory by looking at the ("next", "done") key
    for traj_idx in range(n_trajectories):
        dones = td[traj_idx]["next", "done"].squeeze(-1).to(torch.int64)
        if dones.any():
            done_idx: int = dones.argmax(-1)
        else:
            done_idx = len(dones)
        rewards = td[traj_idx]["next", "reward"].squeeze(-1)[
            : done_idx + 1
        ]  # include the terminal state as well
        returns[traj_idx] = calc_return(rewards, gamma)
    return returns


def normalize_bounded_value(
    tensor: torch.Tensor,
    min_value: float,
    max_value: float,
    new_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Given that we know the minimum and maximum value of all values in `tensor`, normalize them to be in `new_range`."""
    new_min_value, new_max_value = new_range
    if max_value == min_value:
        msg = "min_value and max_value must be different for normalization"
        raise ValueError(msg)
    scale = (new_max_value - new_min_value) / (max_value - min_value)
    return (tensor - min_value) * scale + new_min_value


def add_wandb_key_prefix(prefix: str, key: str) -> str:
    """
    Given a wandb-like logging key like "aaa", "aaa/bbb" or "aaa/bbb.ccc", add a prefix to the last part.

    - "aaa" -> "{prefix}aaa"
    - "aaa/bbb" -> "aaa/{prefix}bbb"
    - "aaa/bbb.ccc" -> "aaa/bbb.{prefix}ccc"
    """
    if "." in key:
        head, tail = key.rsplit(".", 1)
        return f"{head}.{prefix}{tail}"
    if "/" in key:  # there's only ever a single '/'
        head, tail = key.rsplit("/", 1)
        return f"{head}/{prefix}{tail}"
    return f"{prefix}{key}"
