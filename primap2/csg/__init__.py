"""
Composite Source Generator

Generate a composite harmonized dataset from multiple sources according to defined
source priorities and matching algorithms.
"""

from ._compose import compose
from ._models import (
    PriorityDefinition,
    StrategyDefinition,
)
from ._strategies.exceptions import StrategyUnableToProcess
from ._strategies.global_least_squares import GlobalLSStrategy
from ._strategies.substitution import SubstitutionStrategy
from ._wrapper import create_composite_source, create_time_index, set_priority_coords

__all__ = [
    "GlobalLSStrategy",
    "PriorityDefinition",
    "StrategyDefinition",
    "StrategyUnableToProcess",
    "SubstitutionStrategy",
    "compose",
    "create_composite_source",
    "create_time_index",
    "set_priority_coords",
]
