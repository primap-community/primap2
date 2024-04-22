"""Models for the composite source generator."""

import typing
from collections.abc import Hashable

import xarray as xr
from attr import define

from primap2._data_format import ProcessingStepDescription


@define(frozen=True)
class PriorityDefinition:
    """
    Defines source priorities for composing a full dataset or a single timeseries.

    Attributes
    ----------
    priority_dimensions
        List of dimensions from which source timeseries are selected. Each priority has
        to specify all priority dimensions.
    priorities
        List of priority selections. Higher priority selections come first. Each
        selection consists of a dict which maps dimension names to values. Each
        selection has to specify all priority_dimensions, but may specify additional
        dimensions (fixed dimensions) to limit the selection to specific cases.

    Examples
    --------
        [{"area (ISO3)": ["MEX", "COL"], "source": "A"}, {"source": "B"}]
        would select source "A" as highest-priority source and source "B" as
        lower-priority source for Columbia and Mexico, but source "B" as
        highest-priority (and only) source for all other
        countries.
    """

    priority_dimensions: list[Hashable]
    priorities: list[dict[Hashable, str | list[str]]]

    def limit(self, dim: Hashable, value: str) -> "PriorityDefinition":
        """Remove one fixed dimension by limiting to a single value.

        You can't remove priority dimensions, only fixed dimensions.
        """
        new_priorities = []
        for sel in self.priorities:
            if dim not in sel:
                new_priorities.append(sel)
            elif (isinstance(sel[dim], str) and sel[dim] == value) or (
                value in sel[dim]
            ):
                # filter out the matching value
                new_priorities.append({k: v for k, v in sel.items() if k != dim})
        return PriorityDefinition(
            priority_dimensions=self.priority_dimensions, priorities=new_priorities
        )

    def check_dimensions(self):
        """Raise an error if not all priorities specify all priority dimensions."""
        for sel in self.priorities:
            for dim in self.priority_dimensions:
                if dim not in sel:
                    raise ValueError(
                        f"In priority={sel}: missing priority dimension={dim}"
                    )
                if not isinstance(sel[dim], str):
                    raise ValueError(
                        f"In priority={sel}: specified multiple values for priority "
                        f"dimension={dim}, values={sel[dim]}"
                    )


class StrategyUnableToProcess(Exception):
    """The filling strategy is unable to process the given timeseries, possibly due
    to missing data."""

    def __init__(self, reason: str):
        """Specify the reason why the filling strategy is unable to process the data."""
        self.reason = reason


class FillingStrategyModel(typing.Protocol, Hashable):
    """
    Fill missing data in a timeseries using another timeseries.

    You can implement custom filling strategies to use with ``compose`` as long as they
    follow this protocol and are Hashable. To follow the protocol you need to implement
    the ``fill`` method with exactly the parameters and return types as defined below
    and define the ``type`` attribute (see below).

    To ensure that your class is hashable, you have to make sure instances are
    immutable after initialization (this helps caching). The easiest way to ensure
    immutability is usually to use the decorator ``attrs.define`` from the
    ``attrs`` package with the ``frozen=True`` argument.

    Attributes
    ----------
    type
        Short human-readable identifier for your strategy. Avoid special characters
        and spaces.
    """

    type: str

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
    ) -> tuple[xr.DataArray, list[ProcessingStepDescription]]:
        """Fill gaps in ts using data from the fill_ts.

        Using two input timeseries, this builds a composite timeseries and a description
        of the processing steps done. The input timeseries must not be modified.

        Usually, you want to fill missing data (NaNs) in the first timeseries `ts` using
        data from the second timeseries `fill_ts`. However, you are not limited to this,
        you could also check data in `ts` for consistency with data in `fill_ts` and
        discard non-conforming data points so that the resulting timeseries has more
        missing data points.

        Parameters
        ----------
        ts
            Base timeseries. Missing data (NaNs) in this timeseries will be filled or
            other processing is done.
            This function must not modify the data in ts, work on a copy instead.
        fill_ts
            Fill timeseries. Data from this timeseries will be used (possibly after
            modification) to fill missing data in the base timeseries or alter the
            base timeseries in a different form.
            This function must not modify the data in fill_ts.
        fill_ts_repr
            String representation of fill_ts. Human-readable short representation of
            the fill_ts (e.g. the source).

        Returns
        -------
            filled_ts, descriptions. filled_ts contains the result, where missing
            data in ts is (partly) filled using information from fill_ts or other
            processing is done using ts and fill_ts.
            descriptions contains human-readable, structured descriptions of how the
            data was processed, grouped by years for which the same processing steps
            were taken. Every year for which data in filled_ts is different from data
            in ts has to be described and no year for which data was not changed is
            allowed to be described.

        Raises
        ------
        StrategyUnableToProcess
            This exception is raised when the strategy is unable to process the given
            timeseries, possibly due to missing data (e.g. insufficient overlap of
            the two timeseries), bad numerical conditioning or other reasons.
            When this exception is raised, the strategy will be skipped and processing
            continues as if the strategy was not configured for this timeseries, i.e.
            the next applicable filling strategy is used. If no other applicable
            filling strategy is available, an error will be raised.
        """
        ...


@define(frozen=True)
class StrategyDefinition:
    """
    Defines filling strategies for a single timeseries.

    Attributes
    ----------
    strategies
        List of mappings from a timeseries selector to a filling strategy. When a
        timeseries will be used to fill missing data, the list will be checked from the
        start, and the first matching TimeseriesSelector determines the FillingStrategy.
        Example: [({"source": ["FAOSTAT", "UNFCCC]}, StraightStrategy()),
                  ({}, GlobalStrategy())]
        Note that the strategy can depend on fixed dimensions as well as priority
        dimensions.
        In practice, it is usually a good idea to include a default strategy using the
        empty selector {} which matches everything. It has to be the last entry -
        since it matches everything, all entries behind it will be ignored.

    """

    strategies: list[tuple[dict[Hashable, str | list[str]], FillingStrategyModel]]

    def find_strategy(self, fill_ts: xr.DataArray) -> FillingStrategyModel:
        """Find the strategy to use for the given filling timeseries."""
        try:
            return next(self.find_strategies(fill_ts))
        except StopIteration:
            raise KeyError(f"No matching strategy found for {fill_ts.coords}") from None

    def find_strategies(
        self, fill_ts: xr.DataArray
    ) -> typing.Generator[FillingStrategyModel, None, None]:
        """Yields all strategies to use for the timeseries, in configured order."""
        for selector, strategy in self.strategies:
            if self.match(selector=selector, fill_ts=fill_ts):
                yield strategy

    def limit(self, dim: Hashable, value: str) -> "StrategyDefinition":
        """Limit this strategy definition to strategies applicable with the limit."""
        return StrategyDefinition(
            strategies=[
                ({k: v for k, v in sel.items() if k != dim}, strat)
                for (sel, strat) in self.strategies
                if self.match_single_dim(selector=sel, dim=dim, value=value)
            ]
        )

    @staticmethod
    def match(
        *, selector: dict[Hashable, str | list[str]], fill_ts: xr.DataArray
    ) -> bool:
        """Check if a selected timeseries for filling matches the selector."""
        for k, v in selector.items():
            if isinstance(v, str):
                if not fill_ts.coords[k] == v:
                    return False
            elif fill_ts.coords[k] not in v:
                return False
        return True

    @staticmethod
    def match_single_dim(
        *, selector: dict[Hashable, str | list[str]], dim: Hashable, value: str
    ) -> bool:
        """Check if a literal value in one dimension can match the selector."""
        if dim not in selector.keys():
            return True
        if isinstance(selector[dim], str):
            return value == selector[dim]
        return value in selector[dim]
