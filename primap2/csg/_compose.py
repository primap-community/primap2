"""Compose a harmonized dataset from multiple input datasets."""

import xarray as xr
from attrs import define


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
    priorities: list[dict[str, str]]


@define
class FillingStrategy:
    """
    Fill missing data in a dataset using another dataset.
    """

    def fill(self, result_ts: xr.DataArray, fill_ts: xr.DataArray):
        """Fill gaps in the result_da using data from the fill_da.

        The result_ts will be modified, the fill_ts will not be modified.

        The filling may only partly fill missing data.
        """
        ...


@define
class TimeseriesSelector:
    selections: dict[str, str | list[str]]
    """
    Example: [("source", "FAOSTAT"), ("scenario", "high")]
    [("source", ["FAOSTAT", "UNFCCC"])]
    """

    def match(self, fill_ts: xr.DataArray) -> bool:
        """Check if a selected timeseries for filling matches this selector."""
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
        Example: [(("source", ["FAOSTAT", "UNFCCC]), StraightStrategy),
                  ((, ), GlobalStrategy)]
        Note that the strategy can depend on fixed coordinates as well as priority
        coordinates.

    """

    strategies: list[tuple[TimeseriesSelector, FillingStrategy]]


def compose_timeseries(
    input_data: xr.DataArray,
    priority_definition: PriorityDefinition,
    strategy_definition: StrategyDefinition,
) -> xr.DataArray:
    """TODO: logging? source tracebility info?"""
    ...
