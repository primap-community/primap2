"""Compose a harmonized dataset from multiple input datasets."""
from collections.abc import Sequence

import xarray as xr
from attrs import define, frozen


@define
class PriorityDefinition:
    """
    Defines source priorities for a single timeseries.

    Attributes
    ----------
    dimensions
        List of dimensions which are used in the prioritisation.
    priorities
        List of priority selections. Higher priority selections come first. Each
        selection consists of a dict which maps dimension names (which have to be
        in `dimensions`) to values. Example:
        [{"source": "FAOSTAT"}, {"source": "UNFCCC"}]
        would prefer the FAOSTAT data over the UNFCCC data.
        Each selection, applied to the input_data passed, has to yield exactly one
        timeseries.
    """

    dimensions: list[str]
    priorities: list[dict[str, str | int]]


@define
class FillingStrategy:
    """
    Fill missing data in a dataset using another dataset.
    """

    def fill(self, result_da: xr.DataArray, fill_da: xr.DataArray):
        """Fill gaps in the result_da using data from the fill_da.

        The result_da will be modified, the fill_da will not be modified.

        The filling may only partly fill missing data.
        """
        ...


@frozen
class TimeseriesSelector:
    selections: tuple[str, str | int | Sequence[str | int]]

    def match(self, other: dict[str, str | int]) -> bool:
        """Check if a selector as used for xarray's `loc` matches this selector."""
        ...


@define
class StrategyDefinition:
    """
    Defines filling strategies for a single timeseries.

    Attributes
    ----------
    strategies
        List of mappings from a timeseries selector to a filling strategy. When a
        timeseries will be used to fill missing data, the list will be checked from the
        start, and the first matching TimeseriesSelector determines the FillingStrategy.
    """

    strategies: list[tuple[TimeseriesSelector, FillingStrategy]]


def compose_timeseries(
    input_data: xr.DataArray,
    priority_definition: PriorityDefinition,
    strategy_definition: StrategyDefinition,
) -> xr.DataArray:
    ...
