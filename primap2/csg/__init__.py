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
from ._strategies.gaps import FitParameters
from ._strategies.global_least_squares import GlobalLSStrategy
from ._strategies.local_trends import LocalTrendsStrategy
from ._strategies.substitution import SubstitutionStrategy
from ._wrapper import create_composite_source

__all__ = [
    "FitParameters",
    "GlobalLSStrategy",
    "LocalTrendsStrategy",
    "PriorityDefinition",
    "StrategyDefinition",
    "StrategyUnableToProcess",
    "SubstitutionStrategy",
    "compose",
    "create_composite_source",
]
