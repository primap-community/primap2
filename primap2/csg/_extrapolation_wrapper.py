import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from ._models import ExtrapPriorityDefinition, StrategyDefinition
from ._wrapper import create_composite_source

# scenario:
#  we extrapolate different sectors with different data sources and sectors or regions
# this can be mixed. for one source and priority we use parent sector data for the
# next regional data different sectors or countries can have different sources


# TODO:
# * real world example
# * maybe type checks for "entities" etc or own strategy definition with whitelist
# for allowed strategies and configurations?
# finish docstrings


def create_extrapolated_source(
    input_ds: xr.Dataset,
    extrap_priority_definition: ExtrapPriorityDefinition,
    strategy_definition: StrategyDefinition,
    result_prio_coords: dict[str, dict[str, str]],
    limit_coords: dict[str, str | list[str]] | None = None,
    time_range: tuple[str | np.datetime64, str | np.datetime64] | pd.DatetimeIndex | None = None,
    metadata: dict[str, str] | None = None,
    progress_bar: type[tqdm.tqdm] | None = tqdm.tqdm,
) -> xr.Dataset:
    """Create a composite data source including the use of data from different coordinate values.

    This is a wrapper around `primap2.csg.compose` that prepares the input data, sets result
    values for the priority coordinates, and prepares the data for the use of growth rates
    from different coordinate values, e.g. parent categories or regions.

    Parameters
    ----------
    input_ds
        Dataset containing all input data
    extrap_priority_definition
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
        The priority definition also defines the mappings for priorities where
        data from other coordinate values should be used, e.g. regional data or parent
        sector data. The mappings are defined for each priority individually through
        the 'mapping' dict in the priority step.
        Coordinates are the keys and the values are dict[str, str | list],
        where the key is the coordinate value of the data to be used in the current step and
        the value is either a single value or a list of values which define the timeseries
        to be extended.
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
    result_prio_coords
        Defines the values for the priority coordinates in the output dataset. As the
        priority coordinates differ for all input sources there is no canonical value
        for the result and it has to be explicitly defined.
    limit_coords
        Optional parameter to remove data for coordinate values not needed for the
        composition from the input data. The time coordinate is treated separately,
        use ``time_range`` for it. Besides dimensions, the key ``variable`` can be
        used to limit the input data to a subset of the data variables.
    time_range
        Optional parameter to limit the time coverage of the input data.
        Can either be a pandas `DatetimeIndex` or a tuple of `str` or `np.datetime64` in
        the form (year_from, year_to) where both boundaries are included in the range.
        Only the overlap of the supplied index or index created from the tuple with
        the time coordinate of the input dataset will be used.
    metadata
        Set metadata values such as title and references.
    progress_bar
        By default, show progress bars using the tqdm package during the
        operation. If None, don't show any progress bars. You can supply a class
        compatible to tqdm.tqdm's protocol if you want to customize the progress bar.

    Returns
    -------
        xr.Dataset with composed data according to the given priority and strategy
        definitions
    """
    # copy timeseries to the coordinate values they will be used for
    input_ds = _csg_extrapolation_prepare_data(input_ds, extrap_priority_definition)

    # remove extrapolation information from priority definition
    priority_definition = extrap_priority_definition.to_priority_definition()

    result_ds = create_composite_source(
        input_ds,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        result_prio_coords=result_prio_coords,
        limit_coords=limit_coords,
        time_range=time_range,
        metadata=metadata,
        progress_bar=progress_bar,
    )

    return result_ds


def _csg_extrapolation_prepare_data(
    data: xr.Dataset,
    extrap_priority_definition: ExtrapPriorityDefinition,
) -> xr.Dataset:
    # copy time-series for the mapping categories

    # TODO: add processing info after new proc info functionality has been merged
    #  type checking for "entity" has to be a list. if str is allowed in csg
    #  we have to convert to list here before selecting variables

    for prio in extrap_priority_definition.priorities:
        if "mapping" in prio:
            mapping = prio.pop("mapping")
            loc = prio.copy()
            # process loc. All selections have to be lists because
            # else data will be broadcasted along the whole dimension
            for dim in extrap_priority_definition.priority_dimensions:
                if (dim != "mapping") and (isinstance(loc[dim], str)):
                    loc[dim] = [loc[dim]]
            if "variable" in loc:
                variables = loc.pop("variable")
            elif "entity" in loc:
                variables = []
                entity = loc.pop("entity")
                for var in data.data_vars:
                    if data[var].attrs["entity"] in entity:
                        variables.append(var)
            else:
                variables = list(data.data_vars)
            for dim in mapping:
                mapping_dim = mapping[dim]
                for key, value in mapping_dim.items():
                    this_loc = loc.copy()
                    this_loc[dim] = key
                    data_to_copy = data[variables].pr.loc[this_loc]
                    # next: add dim with the same data for all values
                    for var in variables:
                        if isinstance(value, str):
                            data[var] = data[var].pr.set(
                                dim, value, data_to_copy[var], existing="fillna"
                            )
                        else:
                            for val in value:
                                data[var] = data[var].pr.set(
                                    dim, val, data_to_copy[var], existing="fillna"
                                )

    return data
