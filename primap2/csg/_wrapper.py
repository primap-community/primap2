import pandas as pd
import tqdm
import xarray as xr

from ._compose import compose
from ._models import PriorityDefinition, StrategyDefinition


def set_priority_coords(
    ds: xr.Dataset,
    dims: dict[str, dict[str, str]],
) -> xr.Dataset:
    """Set values for priority coordinates.

    Parameters
    ----------
    ds: cr.Dataset
        Dataset to change
    dims: dict
        Dictionary containing coordinate names as keys and as values a dictionary
        with the value to be set and optionally a terminology.
        Examples:
        {"source": {"value": "PRIMAP-hist"}} sets the "source" to "PRIMAP-hist".
        {"area": {"value": "WORLD", "terminology": "ISO3_primap"}} adds the dimension
        "area (ISO3_primap)" to "WORLD".
    """
    for dim in dims.keys():
        terminology = dims[dim].get("terminology", None)
        ds = ds.pr.expand_dims(dim=dim, coord_value=dims[dim]["value"], terminology=terminology)

    return ds


def create_composite_source(
    input_ds: xr.Dataset,
    priority_definition: PriorityDefinition,
    strategy_definition: StrategyDefinition,
    result_prio_coords: dict[str, dict[str, str]],
    limit_coords: dict[str, str | list[str]] | None = None,
    time_range: tuple[str, str] | None = None,
    metadata: dict[str, str] | None = None,
    progress_bar: type[tqdm.tqdm] | None = tqdm.tqdm,
) -> xr.Dataset:
    """Create a composite data source

    This is a wrapper around `primap2.csg.compose` that prepares the input data and sets result
    values for the priority coordinates.


    Parameters
    ----------
    input_ds
        Dataset containing all input data
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
    result_prio_coords
        Defines the vales for the priority coordinates in the output dataset. As the
        priority coordinates differ for all input sources there is no canonical vale
        for the result and it has to be explicitly defined
    limit_coords
        Optional parameter to remove data for coordinate vales not needed for the
        composition from the input data. The time coordinate is treated separately.
    time_range
        Optional parameter to limit the time coverage of the input data. Currently
        only (year_from, year_to) is supported
    metadata
        Set metadata values such as title and references
    progress_bar
        By default, show progress bars using the tqdm package during the
        operation. If None, don't show any progress bars. You can supply a class
        compatible to tqdm.tqdm's protocol if you want to customize the progress bar.

    Returns
    -------
        xr.Dataset with composed data according to the given priority and strategy
        definitions

    """

    # limit input data to these values
    if limit_coords is not None:
        if "variable" in limit_coords.keys():
            variables = limit_coords["variable"]
            limit_coords.pop("variable")
            input_ds = input_ds[variables].pr.loc[limit_coords]

        else:
            input_ds = input_ds.pr.loc[limit_coords]

    # set time range according to input
    if time_range is not None:
        input_ds = input_ds.pr.loc[
            {"time": pd.date_range(time_range[0], time_range[1], freq="YS", inclusive="both")}
        ]

    # run compose
    result_ds = compose(
        input_data=input_ds,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        progress_bar=progress_bar,
    )

    # set priority coordinates
    result_ds = set_priority_coords(result_ds, result_prio_coords)

    if metadata is not None:
        for key in metadata.keys():
            result_ds.attrs[key] = metadata[key]

    result_ds.pr.ensure_valid()

    return result_ds
