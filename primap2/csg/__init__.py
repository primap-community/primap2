"""Composite Source Generator

Generate a composite harmonized dataset from multiple sources according to defined
source priorities and matching algorithms.
"""

from ._compose import compose
from ._models import (
    PriorityDefinition,
    StrategyDefinition,
)
from ._strategies.substitution import SubstitutionStrategy
from ._strategies.global_least_squares import GlobalLSStrategy
from ._strategies.global_least_squares import GlobalLSlstsqStrategy
from ._strategies.null import NullStrategy

__all__ = [
    "compose",
    "PriorityDefinition",
    "StrategyDefinition",
    "SubstitutionStrategy",
    "GlobalLSStrategy",
    "GlobalLSlstsqStrategy",
    "NullStrategy",
]
