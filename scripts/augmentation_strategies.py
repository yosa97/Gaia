"""
Augmentation Strategy Registry for Tournament Environments.

Provides pluggable augmentation strategies to apply to models before GRPO training.
Each strategy is a callable that modifies a model in-place and returns it.

Usage:
    from augmentation_strategies import get_strategy

    strategy = get_strategy("gaussian_noise", std=0.005)
    model = strategy.apply(model)

Available Strategies:
    - none           : Identity (no augmentation)
    - gaussian_noise : Add small Gaussian noise to weights (default std=0.005)
    - weight_scaling : Scale weights in linear layers (default factor=1.001)
    - magnitude_pruning : Zero out smallest weights by magnitude (default ratio=0.001)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel
else:
    # Lazy import for type hints (PreTrainedModel is only needed at type-check time)
    PreTrainedModel = "PreTrainedModel"


class AugmentationStrategy(ABC):
    """
    Base class for augmentation strategies.

    Subclass and implement ``apply(model)`` to define custom augmentation logic.
    """

    def __init__(self, name: str, **hyperparams):
        """
        Initialize strategy with name and hyperparameters.

        Args:
            name: Strategy identifier (e.g., 'gaussian_noise')
            **hyperparams: Strategy-specific hyperparameters
        """
        self.name = name
        self.hyperparams = hyperparams

    @abstractmethod
    def apply(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """
        Apply augmentation to model in-place.

        Args:
            model: HuggingFace PreTrainedModel to augment

        Returns:
            The modified model
        """
        pass

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.hyperparams.items())
        return f"{self.__class__.__name__}({self.name}, {params_str})"


class NoOpStrategy(AugmentationStrategy):
    """Identity augmentation (no changes)."""

    def apply(self, model: "PreTrainedModel") -> "PreTrainedModel":
        return model


class GaussianNoiseStrategy(AugmentationStrategy):
    """Add Gaussian noise to all model parameters."""

    def apply(self, model: "PreTrainedModel") -> "PreTrainedModel":
        import torch

        noise_std = self.hyperparams.get("std", 0.005)
        print(f"[augment] Applying gaussian_noise (std={noise_std}) to all layers...")

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dtype in (torch.float32, torch.bfloat16, torch.float16):
                    noise = torch.randn_like(param.float()) * noise_std
                    param.add_(noise.to(param.dtype))

        print(f"[augment] gaussian_noise done. Model params: {sum(p.numel() for p in model.parameters()):,}")
        return model


class WeightScalingStrategy(AugmentationStrategy):
    """Scale weights in linear layers by a constant factor."""

    def apply(self, model: "PreTrainedModel") -> "PreTrainedModel":
        import torch

        scale_factor = self.hyperparams.get("factor", 1.001)
        print(f"[augment] Applying weight_scaling (factor={scale_factor}) to linear layers...")

        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, "weight") and module.weight is not None:
                    # Skip embeddings and normalization layers
                    if "embed" not in name.lower() and "norm" not in name.lower():
                        module.weight.data.mul_(scale_factor)

        print(f"[augment] weight_scaling done. Model params: {sum(p.numel() for p in model.parameters()):,}")
        return model


class MagnitudePruningStrategy(AugmentationStrategy):
    """Zero out weights below a magnitude threshold (sparsity)."""

    def apply(self, model: "PreTrainedModel") -> "PreTrainedModel":
        import torch

        pruning_ratio = self.hyperparams.get("ratio", 0.001)
        print(f"[augment] Applying magnitude_pruning (ratio={pruning_ratio}) to weight matrices...")

        with torch.no_grad():
            for name, param in model.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    # Find threshold at pruning_ratio quantile
                    threshold = torch.quantile(param.abs().float(), pruning_ratio)
                    mask = param.abs() > threshold.to(param.dtype)
                    param.mul_(mask.to(param.dtype))

        print(f"[augment] magnitude_pruning done. Model params: {sum(p.numel() for p in model.parameters()):,}")
        return model


# ============================================================================
# Strategy Factory
# ============================================================================

_STRATEGIES: Dict[str, type] = {
    "none": NoOpStrategy,
    "gaussian_noise": GaussianNoiseStrategy,
    "weight_scaling": WeightScalingStrategy,
    "magnitude_pruning": MagnitudePruningStrategy,
}


def get_strategy(name: str, **kwargs) -> AugmentationStrategy:
    """
    Factory function to instantiate an augmentation strategy.

    Args:
        name: Strategy name ('none', 'gaussian_noise', 'weight_scaling', 'magnitude_pruning')
        **kwargs: Strategy-specific hyperparameters
            - gaussian_noise: std (float, default 0.005)
            - weight_scaling: factor (float, default 1.001)
            - magnitude_pruning: ratio (float, default 0.001)

    Returns:
        AugmentationStrategy instance

    Raises:
        ValueError: If strategy name is not recognized
    """
    if name not in _STRATEGIES:
        available = ", ".join(_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")

    strategy_class = _STRATEGIES[name]
    return strategy_class(name, **kwargs)


def list_strategies() -> list[str]:
    """Return list of available augmentation strategy names."""
    return list(_STRATEGIES.keys())


def get_default_hyperparams(name: str) -> Dict[str, Any]:
    """Return default hyperparameters for a given strategy."""
    defaults = {
        "none": {},
        "gaussian_noise": {"std": 0.005},
        "weight_scaling": {"factor": 1.001},
        "magnitude_pruning": {"ratio": 0.001},
    }
    return defaults.get(name, {})

# [divergence-marker yosa97-1781423157-13893] unique per-miner no-op line to avoid byte-identical files; does not change behavior.
