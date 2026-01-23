"""Registry pattern for metrics and other extensible components.

This module provides a registry pattern that allows dynamic registration
of metrics, stress tests, and other pluggable components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Generic, Callable

if TYPE_CHECKING:
    from dctt.core.types import Metric

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic registry for components.

    Example usage:
        >>> metric_registry = Registry[Metric]("metrics")
        >>> @metric_registry.register("my_metric")
        ... class MyMetric:
        ...     pass
        >>> metric_registry.get("my_metric")
        <class 'MyMetric'>
    """

    def __init__(self, name: str) -> None:
        """Initialize registry.

        Args:
            name: Human-readable name for error messages.
        """
        self._name = name
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a component.

        Args:
            name: Unique identifier for the component.

        Returns:
            Decorator function.
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(
                    f"Component '{name}' already registered in {self._name} registry"
                )
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Get a registered component by name.

        Args:
            name: Component identifier.

        Returns:
            The registered component class.

        Raises:
            KeyError: If component is not registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"Component '{name}' not found in {self._name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def get_or_none(self, name: str) -> type[T] | None:
        """Get a registered component or None if not found."""
        return self._registry.get(name)

    def list_registered(self) -> list[str]:
        """List all registered component names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._registry

    def __len__(self) -> int:
        """Number of registered components."""
        return len(self._registry)


# Pre-defined registries
MetricRegistry = Registry["Metric"]("metrics")
StressTestRegistry = Registry("stress_tests")
RepairMethodRegistry = Registry("repair_methods")


def register_metric(name: str) -> Callable[[type], type]:
    """Convenience decorator to register a metric.

    Example:
        >>> @register_metric("participation_ratio")
        ... class ParticipationRatio:
        ...     name = "participation_ratio"
        ...     higher_is_worse = False
        ...     def compute(self, embeddings, token_id, neighbors):
        ...         ...
    """
    return MetricRegistry.register(name)


def register_stress_test(name: str) -> Callable[[type], type]:
    """Convenience decorator to register a stress test."""
    return StressTestRegistry.register(name)


def register_repair_method(name: str) -> Callable[[type], type]:
    """Convenience decorator to register a repair method."""
    return RepairMethodRegistry.register(name)
