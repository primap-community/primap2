"""Compose a harmonized dataset from multiple input datasets."""

import math
import typing
from collections.abc import Hashable

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from loguru import logger

from primap2._processing_info import (
    ProcessingStep,
    ProcessingStepDescription,
    _nodes_or_none,
    is_processing_variable,
    processing_variable_name,
)

from . import _models
from ._strategies.exceptions import StrategyUnableToProcess


def compose(
    *,
    input_data: xr.Dataset,
    priority_definition: _models.PriorityDefinition,
    strategy_definition: _models.StrategyDefinition,
    progress_bar: type[tqdm.tqdm] | None = tqdm.tqdm,
) -> xr.Dataset:
    """
    Compose a harmonized dataset from multiple input datasets.

    The input datasets are treated at the timeseries level, for each timeseries:

    * the highest priority dataset is chosen according to the priority_definition
    * if the dataset contains any nan values (denoting missing information), the next
      highest priority dataset is selected to fill the missing values
    * a filling strategy is chosen according to the strategy_definition and the filling
      is done with the dataset selected in the previous step
    * if afterwards there are still missing nan values, the next highest priority
      dataset is selected and the previous step is repeated
    * when all missing values are filled or all input datasets are exhausted, the
      timeseries is finished and written to the result.

    In addition to the harmonized data, also a description of the processing steps
    done for each timeseries is returned in the result dataset, where for each
    variable, a variable of the form "Processing of $variable" is returned, with the
    same dimensions as the variable. It holds the processing step of each data point. If
    input_data has processing information, the steps of the input data points used are
    the parents of the processing steps. Else, the parents are steps without parents
    (function "compose") which name the source of the input timeseries and are created
    for the input data points used.

    Parameters
    ----------
    input_data
        Dataset with the input data. The input data dimensions determine the output
        data dimensions: the output data has the same dimensions as the input data,
        minus the priority dimensions defined in the priority_definition. From the
        priority dimensions, the different datasets for the filling are selected, so
        they vanish in the result.
        The input data is checked using :py:meth:`xarray.Dataset.pr.ensure_valid`
        before processing starts, so that problems in the input data are reported
        directly instead of surfacing later as confusing errors.
    priority_definition
        Defines the priorities to select timeseries from the input data. Priorities
        are formed by a list of selections and are used "from left to right", where the
        first matching selection has the highest priority. Each selection has to specify
        values for all priority dimensions (so that exactly one timeseries is selected
        from the input data), but can also specify other dimensions. That way it is,
        e.g., possible to define a different priority for a specific country by listing
        it early (i.e. with high priority) before the more general rules which should
        be applied for all other countries.
        You can also specify the "entity" or "variable" in the selection, which will
        limit the rule to a specific entity or variable, respectively. For each
        DataArray in the input_data Dataset, the variable is its name, the entity is
        the value of the key `entity` in its attrs.
    strategy_definition
        Defines the filling strategies to be used when filling timeseries with other
        timeseries. Again, the priority is defined by a list of selections and
        corresponding strategies which are used "from left to right". Selections can use
        any dimension and don't have to apply to only one timeseries. For example, to
        define a default strategy which should be used for all timeseries unless
        something else is configured, configure an empty selection as the last
        (rightmost) entry.
        You can also specify the "entity" or "variable" in the selection, which will
        limit the rule to a specific entity or variable, respectively. For each
        DataArray in the input_data Dataset, the variable is its name, the entity is
        the value of the key `entity` in its attrs.
    progress_bar
        By default, show progress bars using the tqdm package during the
        operation. If None, don't show any progress bars. You can supply a class
        compatible to tqdm.tqdm's protocol if you want to customize the progress bar.

    Returns
    -------
        result
            Dataset with the same entities and dimensions as input_data, but with
            following changes: the data is composed and filled according to the rules,
            the priority dimensions are reduced and not included in the result, and
            additional variables of the form "Processing of $variable" are added which
            describe the processing steps done for each data point.
    """
    input_data.pr.ensure_valid()

    result_das = {}
    input_data = input_data.pr.dequantify()
    variables = [var for var in input_data.data_vars if not is_processing_variable(var)]

    if progress_bar is None:
        variable_iterator = variables
    else:
        variable_iterator = progress_bar(variables)

    priority_dimensions = priority_definition.priority_dimensions
    priority_definition.check_dimensions()

    strategy_definition.check_dimensions(input_data)

    for variable in variable_iterator:
        if progress_bar is not None:
            variable_iterator.set_postfix_str(str(variable))

        input_da = input_data[variable]
        entity = input_da.attrs["entity"]
        input_processing_da = input_data.get(processing_variable_name(variable))

        # all dimensions are either time, priority selection dimensions, or need to
        # be iterated over
        group_by_dimensions = tuple(
            dim for dim in input_da.dims if dim != "time" and dim not in priority_dimensions
        )

        (
            result_das[variable],
            result_das[processing_variable_name(variable)],
        ) = preallocate_result_arrays(
            input_da=input_da,
            group_by_dimensions=group_by_dimensions,
            priority_dimensions=priority_dimensions,
        )

        number_of_timeseries = math.prod(len(input_da[dim]) for dim in group_by_dimensions)

        if progress_bar is None:
            pbar = None
        else:
            pbar = progress_bar(total=number_of_timeseries, unit="ts", unit_scale=True)

        iterate_next_fixed_dimension(
            input_da=input_da,
            priority_definition=priority_definition.limit("entity", entity).limit(
                "variable", variable
            ),
            strategy_definition=strategy_definition.limit("entity", entity).limit(
                "variable", variable
            ),
            group_by_dimensions=group_by_dimensions,
            result_da=result_das[variable],
            result_processing_da=result_das[processing_variable_name(variable)],
            progress_bar=pbar,
            input_processing_da=input_processing_da,
        )
        if pbar is not None:
            pbar.close()

    result_ds = xr.Dataset(result_das).pr.quantify()
    # composing removes the priority dimensions, also remove the attrs describing
    # the priority dimensions
    result_ds.attrs = input_data.attrs.copy()
    for dim_key in priority_dimensions:
        if isinstance(dim_key, str) and "(" in dim_key:
            dim = dim_key.split("(", 1)[0][:-1]
        else:
            dim = dim_key
        if dim == "area":
            del result_ds.attrs["area"]
        elif dim == "category":
            del result_ds.attrs["cat"]
        elif dim == "scenario":
            del result_ds.attrs["scen"]
    return result_ds


def preallocate_result_arrays(
    *,
    group_by_dimensions: typing.Iterable[Hashable],
    input_da: xr.DataArray,
    priority_dimensions: typing.Iterable[Hashable],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Create empty arrays in the right shape to hold the result."""
    result_da = xr.DataArray(
        data=np.nan,
        dims=["time", *group_by_dimensions],
        coords={
            dim: input_da.coords[dim] for dim in input_da.coords if dim not in priority_dimensions
        },
        attrs=input_da.attrs,
        name=input_da.name,
    )
    processing_result_da = xr.DataArray(
        data=np.full(result_da.shape, None, dtype=object),
        dims=result_da.dims,
        coords=result_da.coords,
        attrs={
            "entity": processing_variable_name(input_da.name),
            "described_variable": input_da.name,
        },
    )
    return result_da, processing_result_da


def priority_coordinates_repr(*, fill_ts: xr.DataArray, priority_dimensions: list[Hashable]) -> str:
    """Reduce the priority coordinates to a short string representation."""
    priority_coordinates: dict[str, str] = {str(k): fill_ts[k].item() for k in priority_dimensions}
    if len(priority_coordinates) == 1:
        # only one priority dimension, just output the value because it is clear what is
        # meant
        return repr(next(iter(priority_coordinates.values())))
    return repr(priority_coordinates)


def _select(da: xr.DataArray | None, selector: dict[Hashable, typing.Any]) -> xr.DataArray | None:
    return None if da is None else da.loc[selector]


def iterate_next_fixed_dimension(
    *,
    input_da: xr.DataArray,
    priority_definition: _models.PriorityDefinition,
    strategy_definition: _models.StrategyDefinition,
    group_by_dimensions: tuple[Hashable, ...],
    result_da: xr.DataArray,
    result_processing_da: xr.DataArray,
    progress_bar: tqdm.tqdm | None,
    input_processing_da: xr.DataArray | None = None,
    coords: dict[str, typing.Any] | None = None,
) -> None:
    """Recursively iterate over dimensions in group_by_dimensions.

    If there is only one dimension left, actually compute results and store
    them in the result_da and the result_processing_da. Otherwise, iterate over one
    dimension and recursively call iterate_next_fixed_dimension again.
    input_processing_da is the optional processing information of input_da, coords are
    the values of the dimensions already iterated over.
    """
    my_dim = group_by_dimensions[0]
    new_group_by_dimensions = group_by_dimensions[1:]
    for val_array in input_da[my_dim]:
        val = val_array.item()
        fixed_coords = {**(coords or {}), str(my_dim): val}
        limited_strategy_definition = strategy_definition.limit(dim=my_dim, value=val)
        limited_priority_definition = priority_definition.limit(dim=my_dim, value=val)
        if new_group_by_dimensions:
            # have to iterate further until all dimensions are consumed
            iterate_next_fixed_dimension(
                input_da=input_da.loc[{my_dim: val}],
                priority_definition=limited_priority_definition,
                strategy_definition=limited_strategy_definition,
                group_by_dimensions=new_group_by_dimensions,
                result_da=result_da.loc[{my_dim: val}],
                result_processing_da=result_processing_da.loc[{my_dim: val}],
                progress_bar=progress_bar,
                input_processing_da=_select(input_processing_da, {my_dim: val}),
                coords=fixed_coords,
            )
        else:
            # Result exclusions are handled here (per definition, we don't do any
            # processing on result exclusions) but input data exclusions are handled
            # in compose_timeseries (per definition, we skip to the next source when
            # a source is skipped due to input data exclusions).
            if not limited_priority_definition.excludes_result(result_da.loc[{my_dim: val}]):
                # actually compute results
                (
                    result_da.loc[{my_dim: val}],
                    result_processing_da.loc[{my_dim: val}],
                ) = compose_timeseries(
                    input_data=input_da.loc[{my_dim: val}],
                    priority_definition=limited_priority_definition,
                    strategy_definition=limited_strategy_definition,
                    input_processing=_select(input_processing_da, {my_dim: val}),
                    coords=fixed_coords,
                )
            if progress_bar is not None:
                progress_bar.update()


def compose_timeseries(
    *,
    input_data: xr.DataArray,
    priority_definition: _models.PriorityDefinition,
    strategy_definition: _models.StrategyDefinition,
    input_processing: xr.DataArray | None = None,
    coords: dict[str, typing.Any] | None = None,
) -> tuple[xr.DataArray, np.ndarray]:
    """
    Compute a single timeseries from given input data, priorities, and strategies.

    Parameters
    ----------
    input_data
        The input data which has dimensions time plus the priority dimensions. The
        fixed coordinates are supplied as zero-dimensional coordinates.
    input_processing
        Optional: the processing information of input_data. The processing steps of the
        result have the steps of the input data points used as parents.
    coords
        Optional: the coordinates of the result, which are recorded in its processing
        steps.
    priority_definition
        The definition of priorities within input_data. Each priority selects a single
        timeseries (i.e. an array with only time as the dimension), so has to specify
        values for all priority dimensions.
    strategy_definition
        The definition of strategies for timeseries in input_data.

    Returns
    -------
        result_ts, processing_steps. In result_ts is the numerical result, with
        the time as the only dimension.
        processing_steps is an object array over time with the processing step which
        set each data point of the result, or None for data points without data.
    """
    context_logger = logger.bind(
        fixed_coordinates={k: v for k, v in input_data.coords.items() if v.shape == ()},
        priority_coordinates={
            k: list(v.data) for k, v in input_data.coords.items() if v.shape != ()
        },
        priorities=priority_definition.priorities,
        strategies=strategy_definition.strategies,
    )

    result_ts: xr.DataArray | None = None
    tracker = _StepTracker(input_data=input_data, input_processing=input_processing, coords=coords)

    for selector in priority_definition.priorities:
        try:
            fill_ts = input_data.loc[selector]
        except KeyError:
            context_logger.debug(f"{selector=} matched no input_data, skipping.")
            continue
        if input_processing is None:
            fill_steps = [None] * len(tracker.time_points)
        else:
            fill_steps = _nodes_or_none(input_processing.loc[selector].values).tolist()

        fill_ts_repr = priority_coordinates_repr(
            fill_ts=fill_ts,
            priority_dimensions=priority_definition.priority_dimensions,
        )
        # remove priority dimension information, it messes with automatic alignment
        # in computations. The corresponding information is now in fill_ts_repr.
        fill_ts_no_prio_dims = fill_ts.drop_vars(priority_definition.priority_dimensions)

        if result_ts is None:
            result_ts = xr.full_like(fill_ts_no_prio_dims, np.nan)

        if priority_definition.excludes_input(fill_ts):
            context_logger.debug(f"{fill_ts_repr} is excluded from processing, skipped.")
            continue
        if fill_ts.isnull().all():
            context_logger.debug(f"{fill_ts_repr} is fully NaN, skipped.")
            continue

        context_logger.debug(f"Filling with {fill_ts_repr} now.")

        for strategy in strategy_definition.find_strategies(fill_ts):
            try:
                result_ts, steps = strategy.fill(
                    ts=result_ts,
                    fill_ts=fill_ts_no_prio_dims,
                    fill_ts_repr=fill_ts_repr,
                )
                new_values = result_ts.values
                for step in steps:
                    if len(step.time):
                        tracker.apply_step(step, new_values, fill_steps, fill_ts_repr)
                    else:
                        context_logger.debug(
                            f"No data changed using {fill_ts_repr}: {step.description}"
                        )
                break
            except StrategyUnableToProcess as err:
                context_logger.info(
                    f"Strategy {strategy.type} unable to process {fill_ts_repr} "
                    f"({err.reason}), skipping to next strategy."
                )
                # next strategy
                continue
        else:
            # only executed if there was no `break` in the for loop, i.e. if no strategy
            # was applicable and worked
            raise ValueError(
                f"No configured strategy was able to process "
                f"\n{input_data.coords}\n{input_data.attrs}\n{strategy_definition=}"
            )

        if not result_ts.isnull().any():
            context_logger.debug("No NaNs remaining, skipping the rest of the sources.")
            break

    if result_ts is None:
        raise ValueError(
            f"No priority selector matched for "
            f"\n{input_data.coords}\n{input_data.attrs}\n{priority_definition=}"
        )

    processing_steps = np.empty(len(tracker.result_steps), dtype=object)
    processing_steps[:] = tracker.result_steps
    return result_ts, processing_steps


class _StepTracker:
    """The processing steps of the data points of a timeseries during composing."""

    def __init__(
        self,
        *,
        input_data: xr.DataArray,
        input_processing: xr.DataArray | None,
        coords: dict[str, typing.Any] | None,
    ):
        self.variable = input_data.name
        self.coords = coords
        self.untracked_input = input_processing is None
        time_values = input_data["time"].values
        self.time_points = list(time_values)
        self.time_index = pd.Index(time_values)
        # the current processing step of each data point of the result
        self.result_steps: list[ProcessingStep | None] = [None] * len(self.time_points)
        self.leaf_descriptions: dict[str, ProcessingStepDescription] = {}

    def fill_step_at(self, i: int, fill_steps: list[ProcessingStep | None], fill_ts_repr: str):
        """The processing step of the input data point, for untracked input a leaf which
        is created when it is first used."""
        if fill_steps[i] is None and self.untracked_input:
            if fill_ts_repr not in self.leaf_descriptions:
                self.leaf_descriptions[fill_ts_repr] = ProcessingStepDescription(
                    function="compose",
                    description=f"input data {fill_ts_repr}",
                    source=fill_ts_repr,
                    variable=self.variable,
                    coords=self.coords,
                )
            fill_steps[i] = ProcessingStep(
                self.leaf_descriptions[fill_ts_repr], self.time_points[i]
            )
        return fill_steps[i]

    def apply_step(
        self,
        step: _models.StrategyStep,
        new_values: np.ndarray,
        fill_steps: list[ProcessingStep | None],
        fill_ts_repr: str,
    ) -> None:
        """Set the processing steps of the data points changed by the strategy step."""
        description = ProcessingStepDescription(
            function=step.function,
            description=step.description,
            variable=self.variable,
            coords=self.coords,
        )
        changed = self.time_index.get_indexer(step.time)
        if (changed < 0).any():
            raise ValueError(f"Strategy step {description.function!r} changed unknown time points.")
        # steps of inputs which are the same for all changed data points
        common = _unique_steps(
            [self.result_steps[i] for i in self.time_index.get_indexer(step.ts_time)]
            + [
                self.fill_step_at(i, fill_steps, fill_ts_repr)
                for i in self.time_index.get_indexer(step.fill_ts_time)
            ]
        )
        common_ids = {id(x) for x in common}
        new_steps: list[ProcessingStep | None] = []
        for i in changed.tolist():
            if np.isnan(new_values[i]):
                new_steps.append(None)
                continue
            fill_step = self.fill_step_at(i, fill_steps, fill_ts_repr)
            if fill_step is None:
                parents = common
            elif id(fill_step) in common_ids:
                parents = (fill_step, *(x for x in common if x is not fill_step))
            else:
                parents = (fill_step, *common)
            new_steps.append(ProcessingStep(description, self.time_points[i], parents))
        # after calculating all, as the inputs are the steps before this step
        for i, new_step in zip(changed.tolist(), new_steps, strict=True):
            self.result_steps[i] = new_step


def _unique_steps(steps: list[ProcessingStep | None]) -> tuple[ProcessingStep, ...]:
    """The steps which are not None without duplicates, keeping the order."""
    return tuple({id(x): x for x in steps if x is not None}.values())
