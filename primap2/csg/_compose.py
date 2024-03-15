"""Compose a harmonized dataset from multiple input datasets."""

import typing

import numpy as np
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
    Fill missing data in a timeseries using another timeseries.
    """

    type: str

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
        sources_ts: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Fill gaps in ts using data from the fill_ts.

        Parameters
        ----------
        ts
            Base timeseries. Missing data (NaNs) in this timeseries will be filled.
            This function must not modify the data in ts.
        fill_ts
            Fill timeseries. Data from this timeseries will be used (possibly after
            modification) to fill missing data in the base timeseries.
            This function must not modify the data in fill_ts.
        fill_ts_repr
            String representation of fill_ts. Human-readable short representation of
            the fill_ts (e.g. the source).
        sources_ts
            Source information timeseries. String representation of the sources
            for the data in ts, with the same shape as ts.
            This function must not modify the data in sources_ts.

        Returns
        -------
            filled_ts, filled_sources_ts. filled_ts contains the result, where missing
            data in ts is (partly) filled using information from fill_ts.
            filled_sources_ts contains a string representation of the derivation for
            each data point.
        """
        ...


class SubstitutionStrategy:
    """Fill missing data in the result dataset by copying.

    The NaNs in the result dataset are substituted with data from the filling
    dataset.
    """

    type = "substitution"

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
        sources_ts: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Fill gaps in ts using data from the fill_ts.

        Parameters
        ----------
        ts
            Base timeseries. Missing data (NaNs) in this timeseries will be filled.
            This function does not modify the data in ts.
        fill_ts
            Fill timeseries. Data from this timeseries will be used (possibly after
            modification) to fill missing data in the base timeseries.
            This function does not modify the data in fill_ts.
        fill_ts_repr
            String representation of fill_ts. Human-readable short representation of
            the fill_ts (e.g. the source).
        sources_ts
            Source information timeseries. String representation of the sources
            for the data in ts, with the same shape as ts.
            This function does not modify the data in sources_ts.

        Returns
        -------
            filled_ts, filled_sources_ts. filled_ts contains the result, where missing
            data in ts is (partly) filled using unmodified data from fill_ts.
            filled_sources_ts is the same as sources_ts, but with fill_ts_repr at
            coordinates which were filled in filled_ts.
        """
        # data
        filled_ts = xr.core.ops.fillna(ts, fill_ts, join="exact")
        # source info
        fill_sources_ts = xr.full_like(fill_ts, fill_ts_repr, dtype=object)
        fill_sources_ts[fill_ts.isnull()] = np.nan
        filled_sources_ts = xr.core.ops.fillna(
            sources_ts, fill_sources_ts, join="exact"
        )

        return filled_ts, filled_sources_ts


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


def priority_coordinates_repr(
    *, fill_ts: xr.DataArray, priority_dimensions: list[str]
) -> str:
    """Reduce the priority coordinates to a short string representation"""
    priority_coordinates: dict[str, str] = {
        k: fill_ts[k].item() for k in priority_dimensions
    }
    if len(priority_coordinates) == 1:
        # only one priority dimension, just output the value because it is clear what is
        # meant
        return repr(next(iter(priority_coordinates.values())))
    return repr(priority_coordinates)


def compose_timeseries(
    *,
    input_data: xr.DataArray,
    priority_definition: PriorityDefinition,
    strategy_definition: StrategyDefinition,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute a single timeseries from given input data, priorities, and strategies.

    Parameters
    ----------
    input_data
        The input data which has dimensions time plus the priority dimensions. The
        fixed coordinates are supplied as zero-dimensional coordinates.
    priority_definition
        The definition of priorities within input_data. Each priority selects a single
        timeseries (i.e. an array with only time as the dimension), so has to specify
        values for all priority dimensions.
    strategy_definition
        The definition of strategies for timeseries in input_data.

    Returns
    -------
        result_ts, result_sources_ts. In result_ts is the numerical result, in
        result_sources_ts is the string representation of the derivation of each data
        point. Both are a single timeseries with the same fixed coordinates as
        input_data and the same time dimension like input_data.
    """
    context_logger = logger.bind(
        fixed_coordinates={k: v for k, v in input_data.coords.items() if v.shape == ()},
        priority_coordinates={
            k: list(v.data) for k, v in input_data.coords.items() if v.shape != ()
        },
        priorities=priority_definition.priorities,
        strategies=strategy_definition.strategies,
    )

    result_ts: None | xr.DataArray = None
    result_sources_ts: None | xr.DataArray = None
    for selector in priority_definition.priorities:
        try:
            fill_ts = input_data.loc[selector]
        except KeyError:
            context_logger.debug(f"{selector=} matched no input_data, skipping.")
            continue

        fill_ts_repr = priority_coordinates_repr(
            fill_ts=fill_ts, priority_dimensions=priority_definition.dimensions
        )

        if result_ts is None or result_sources_ts is None:
            context_logger.debug(
                f"{fill_ts_repr} is the highest-priority source, using as the "
                f"basis to fill."
            )
            result_ts = fill_ts
            result_sources_ts = xr.full_like(
                fill_ts, fill_value=fill_ts_repr, dtype=object
            )
            result_sources_ts[result_ts.isnull()] = np.nan
        else:
            context_logger.debug(f"Filling with {fill_ts_repr} now.")
            strategy = strategy_definition.find_strategy(fill_ts)
            result_ts, result_sources_ts = strategy.fill(
                ts=result_ts,
                fill_ts=fill_ts,
                fill_ts_repr=fill_ts_repr,
                sources_ts=result_sources_ts,
            )

        if not result_ts.isnull().any():
            context_logger.debug("No NaNs remaining, skipping the rest of the sources.")
            break

    if result_ts is None:
        raise ValueError(
            f"No selector matched for \n{input_data.coords}\n{priority_definition=}"
        )

    return result_ts, result_sources_ts
