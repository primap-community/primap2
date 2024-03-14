"""Compose a harmonized dataset from multiple input datasets."""

import typing

import xarray as xr
from attrs import define
from loguru import logger


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


class FillingStrategyModel(typing.Protocol):
    """
    Fill missing data in a dataset using another dataset.
    """

    type: str

    def fill(self, ts: xr.DataArray, fill_ts: xr.DataArray) -> xr.DataArray:
        """Fill gaps in ts using data from the fill_ts.

        ts and fill_ts will not be modified.

        The filling may only partly fill missing data.
        """
        ...


class SubstitutionStrategy:
    """Fill missing data in the result dataset by copying.

    The NaNs in the result dataset are substituted with data from the filling
    dataset.
    """

    type = "substitution"

    def fill(self, ts: xr.DataArray, fill_ts: xr.DataArray) -> xr.DataArray:
        """Fill gaps in ts using data from the fill_ts."""
        return xr.core.ops.fillna(ts, fill_ts, join="exact")


@define
class TimeseriesSelector:
    selections: dict[str, str | list[str]]
    """
    Examples:
    [("source", "FAOSTAT"), ("scenario", "high")]
    [("source", ["FAOSTAT", "UNFCCC"])]
    """

    def match(self, fill_ts: xr.DataArray) -> bool:
        """Check if a selected timeseries for filling matches this selector."""
        return all(fill_ts.coords[k] == v for (k, v) in self.selections.items())


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

    strategies: list[tuple[TimeseriesSelector, FillingStrategyModel]]

    def find_strategy(self, fill_ts: xr.DataArray) -> FillingStrategyModel:
        """Find the strategy to use for the given filling timeseries."""
        for selector, strategy in self.strategies:
            if selector.match(fill_ts):
                return strategy
        raise KeyError(f"No matching strategy found for {fill_ts.coords}")


def compose_timeseries(
    *,
    input_data: xr.DataArray,
    priority_definition: PriorityDefinition,
    strategy_definition: StrategyDefinition,
) -> xr.DataArray:
    """TODO: logging? source tracebility info?"""
    context_logger = logger.bind(
        fixed_coordinates={k: v for k, v in input_data.coords.items() if v.shape == ()},
        priority_coordinates={
            k: list(v.data) for k, v in input_data.coords.items() if v.shape != ()
        },
        priorities=priority_definition.priorities,
        strategies=strategy_definition.strategies,
    )

    result_ts: None | xr.DataArray = None
    for selector in priority_definition.priorities:
        try:
            fill_ts = input_data.loc[selector]
        except KeyError:
            context_logger.debug(f"{selector=} matched no input_data, skipping.")
            continue

        if result_ts is None:
            context_logger.debug(
                f"{ {k: fill_ts[k].item() for k in priority_definition.dimensions} } is"
                f" the highest-priority source, using as the basis to fill."
            )
            result_ts = fill_ts
        else:
            strategy = strategy_definition.find_strategy(fill_ts)
            result_ts = strategy.fill(result_ts, fill_ts)

        if not result_ts.isnull().any():
            context_logger.debug("No NaNs remaining, skipping the rest of the sources.")
            break

    if result_ts is None:
        raise ValueError(
            f"No selector matched for \n{input_data.coords}\n{priority_definition=}"
        )

    return result_ts
