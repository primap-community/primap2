"""Processing information for the data points of a data variable."""

import array
import functools
import inspect
import json
import operator
import typing
import uuid

import numpy as np
import pandas as pd
import xarray as xr
from attr import define, evolve, field
from loguru import logger

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor

PROCESSING_PREFIX = "Processing of "


def is_processing_variable(name: typing.Hashable) -> bool:
    """True if the variable of this name carries processing information."""
    return isinstance(name, str) and name.startswith(PROCESSING_PREFIX)


def processing_variable_name(described_variable: typing.Hashable) -> str:
    """The name of the variable which carries the processing information for a variable.

    Parameters
    ----------
    described_variable
        The name of the variable whose processing information is described.
    """
    return f"{PROCESSING_PREFIX}{described_variable}"


def has_processing_info(ds: xr.Dataset) -> bool:
    """True if the dataset carries processing information for at least one variable."""
    return any(is_processing_variable(var) for var in ds)


def ensure_complete_processing_info(ds: xr.Dataset) -> None:
    """Raise if the processing information of the dataset doesn't match its data.

    If a dataset has processing information, every data variable needs processing
    information. A data point has a processing step if and only if it has data, and the
    time of the step is the time of the data point.

    Raises
    ------
    ValueError
        If processing information is missing for a data variable or for a data point
        with data, if there is processing information for a data point without data, or
        if the time of a step doesn't match the data point.
    """
    if not has_processing_info(ds):
        return
    for var in ds.data_vars:
        if is_processing_variable(var):
            continue
        name = processing_variable_name(var)
        if name not in ds:
            raise ValueError(f"The dataset has processing information, but not for {var!r}.")
        processing = ds[name]
        da = ds[var]
        if set(processing.dims) != set(da.dims):
            raise ValueError(
                f"The processing information of {var!r} has the dimensions "
                f"{_dims_repr(processing.dims)}, but needs {_dims_repr(da.dims)}."
            )
        has_data = da.notnull().transpose(*processing.dims).values
        has_node = ~pd.isnull(processing.values)
        missing = has_data & ~has_node
        if missing.any():
            index = tuple(np.argwhere(missing)[0])
            coordinates = _coordinates_repr(_index_coordinates(processing, index))
            raise ValueError(
                f"{int(missing.sum())} data points of {var!r} contain data, but have no "
                f"processing information, e.g. {coordinates}."
            )
        superfluous = has_node & ~has_data
        if superfluous.any():
            index = tuple(np.argwhere(superfluous)[0])
            coordinates = _coordinates_repr(_index_coordinates(processing, index))
            raise ValueError(
                f"{int(superfluous.sum())} data points of {var!r} have processing information, "
                f"but contain no data, e.g. {coordinates}."
            )
        _ensure_node_times(var, processing, has_node)


def _ensure_node_times(var: typing.Hashable, processing: xr.DataArray, has_node: np.ndarray):
    """Raise if the time of a step differs from the time of its data point."""
    nodes = processing.values[has_node]
    try:
        node_times = np.fromiter(
            map(operator.attrgetter("time"), nodes), dtype="datetime64[ns]", count=len(nodes)
        )
    except AttributeError as err:
        raise ValueError(
            f"The processing information of {var!r} contains something other than processing steps."
        ) from err
    axis = processing.dims.index("time")
    times = processing["time"].values.astype("datetime64[ns]")
    shape = [1] * processing.ndim
    shape[axis] = -1
    point_times = np.broadcast_to(times.reshape(shape), processing.shape)[has_node]
    wrong = node_times != point_times
    if wrong.any():
        position = int(np.flatnonzero(wrong)[0])
        index = tuple(np.argwhere(has_node)[position])
        coordinates = _coordinates_repr(_index_coordinates(processing, index))
        raise ValueError(
            f"{int(wrong.sum())} processing steps of {var!r} have a different time than their "
            f"data point, e.g. {coordinates}."
        )


_PROCESSING_STEP_HINT = (
    "wrapping the call in a ds.pr.processing_step() block, which records it as one "
    "processing step of the dataset"
)


def ensure_complete_processing_info_of_other(
    ds: xr.Dataset, other: xr.Dataset | xr.DataArray
) -> None:
    """Raise if ``ds`` has processing information, but ``other``, from which data is taken
    for ``ds``, has not for every timeseries.

    Raises
    ------
    TypeError
        If ``ds`` has processing information and ``other`` is a DataArray.
    ValueError
        If ``ds`` has processing information and ``other`` has incomplete processing
        information.
    """
    if not has_processing_info(ds):
        return
    if isinstance(other, xr.DataArray):
        raise TypeError(
            "The dataset has processing information, but the DataArray it is combined with "
            "can not carry any. Consider creating a dataset with processing information "
            "from it with da.pr.with_creation_step(), which is then only combined with the "
            "variable of the same name, or wrapping the call in a ds.pr.processing_step() "
            "block, which combines the DataArray with all variables and records the call "
            "as one processing step of the dataset."
        )
    if not has_processing_info(other):
        raise ValueError(
            "The dataset has processing information, but the data it is combined with has "
            "none, so the history of the result would be unknown. Consider adding "
            "processing information to the other data, or " + _PROCESSING_STEP_HINT + "."
        )
    try:
        ensure_complete_processing_info(other)
    except ValueError as err:
        raise ValueError(
            f"The processing information of the data it is combined with is incomplete: "
            f"{err} Consider " + _PROCESSING_STEP_HINT + "."
        ) from err


def ensure_no_processing_info(ds: xr.Dataset) -> None:
    """Raise if the dataset carries processing information.

    Use this in functions which would silently drop or corrupt the processing
    information of their input, so that users have to explicitly discard it instead.

    Parameters
    ----------
    ds
        The dataset to check.

    Raises
    ------
    NotImplementedError
        If the dataset contains processing information for at least one variable.
    """
    if has_processing_info(ds):
        raise NotImplementedError(
            "Dataset contains processing information, this is not supported yet. "
            "Use ds.pr.without_processing_info()."
        )


@define(frozen=True, kw_only=True)
class ProcessingStepDescription:
    """Description of a processing step done on a timeseries.

    One description is shared by all data points of a timeseries which were set by
    the step, see :py:class:`ProcessingStep`.

    Attributes
    ----------
    id
        Unique identifier of the description.
    function
        The name of the function which did the processing.
    description
        Human-readable description of the processing step.
    source
        Optional: a short identifier for the source of the data which was used for the
        processing.
    variable
        Optional: the name of the data variable of the processed timeseries.
    coords
        Optional: the coordinates of the processed timeseries at the time of the
        processing, apart from "time".
    """

    id: uuid.UUID = field(factory=uuid.uuid4)
    function: str
    description: str
    source: str | None = None
    variable: str | None = None
    coords: dict[str, typing.Any] | None = field(
        default=None, converter=lambda x: None if x is None else dict(x), hash=False
    )


def _format_time(time: np.datetime64) -> str:
    """Year if the time is January 1st at midnight, else the date (or the full time)."""
    day = time.astype("datetime64[D]")
    if day.astype(time.dtype) != time:
        return str(time)
    if day == day.astype("datetime64[Y]").astype("datetime64[D]"):
        return str(day.astype("datetime64[Y]"))
    return str(day)


@define(frozen=True)
class ProcessingStep:
    """A processing step which set the value of one data point.

    Steps form the history of a data point: every step refers to the steps which set
    the values it was computed from as its parents, which may be at other time points.

    Equality and hash only consider ``description`` and ``time``, never the parents:
    a description has exactly one set of parents per time point.

    Attributes
    ----------
    description
        The description of the step, shared with the other data points of the timeseries
        set by the step.
    time
        The time point of the data point set by the step.
    parents
        The steps which set the values this value was computed from. Empty if the step
        created the value.
    """

    description: ProcessingStepDescription
    time: np.datetime64
    parents: tuple["ProcessingStep", ...] = field(default=(), converter=tuple, eq=False, repr=False)

    def __str__(self) -> str:
        d = self.description
        text = d.function
        if d.source is not None:
            text += f" (data source: {d.source})"
        text += f" for {_format_time(self.time)}: {d.description}"
        timeseries = " ".join(
            x
            for x in (d.variable, None if not d.coords else _coordinates_repr(d.coords))
            if x is not None
        )
        return f"{timeseries}: {text}" if timeseries else text

    def history(self) -> list["ProcessingStep"]:
        """All processing steps which led to this step, including the step itself."""
        ordered: list[ProcessingStep] = []
        visited: set[int] = set()
        # depth-first search without recursion, so that long histories don't exhaust
        # the stack. Steps are marked as "expanded" once their parents are on the stack.
        stack: list[tuple[ProcessingStep, bool]] = [(self, False)]
        while stack:
            step, expanded = stack.pop()
            if expanded:
                ordered.append(step)
                continue
            if id(step) in visited:
                continue
            visited.add(id(step))
            stack.append((step, True))
            stack.extend((parent, False) for parent in reversed(step.parents))
        return ordered

    def format_history(self) -> str:
        """Human-readable tree of all processing steps which led to this step.

        The first line is this step, followed by its parents. A single parent follows
        marked with "↑" and without indentation, several parents are shown as indented
        branches. Steps which are reached via several branches are labelled like "(a)"
        at their first occurrence, later occurrences only refer to the label like
        "see (a)".
        """
        references: dict[int, int] = {}
        for step in self.history():
            for parent in step.parents:
                references[id(parent)] = references.get(id(parent), 0) + 1
        labels: dict[int, str] = {}

        lines = []
        # depth-first search without recursion, see history(). Besides the step, the
        # stack holds the prefix of its line, the prefix for a single parent and the
        # prefix for branches, which start below the text of the step.
        stack: list[tuple[ProcessingStep, str, str, str]] = [(self, "", "", "")]
        while stack:
            step, prefix, chain_prefix, branch_prefix = stack.pop()
            if id(step) in labels:
                lines.append(f"{prefix}see ({labels[id(step)]})")
                continue
            if references.get(id(step), 0) > 1:
                labels[id(step)] = _label(len(labels))
                prefix += f"({labels[id(step)]}) "
            lines.append(f"{prefix}{step}")
            if len(step.parents) == 1:
                stack.append(
                    (step.parents[0], chain_prefix + "↑ ", chain_prefix, chain_prefix + "  ")
                )
                continue
            branches = []
            for i, parent in enumerate(step.parents):
                last = i == len(step.parents) - 1
                continuation = branch_prefix + ("   " if last else "│  ")
                branches.append(
                    (
                        parent,
                        branch_prefix + ("└─ " if last else "├─ "),
                        continuation,
                        continuation,
                    )
                )
            stack.extend(reversed(branches))
        return "\n".join(lines)


def _label(index: int) -> str:
    """The label with the given index in "a" to "z", "aa" to "zz", "aaa" and so on."""
    # skip the labels which are shorter than the result
    length = 1
    while index >= 26**length:
        index -= 26**length
        length += 1
    letters = []
    for _ in range(length):
        index, remainder = divmod(index, 26)
        letters.append(chr(ord("a") + remainder))
    return "".join(reversed(letters))


def _coordinate_lists(template: xr.DataArray, dims: list[typing.Hashable]) -> list[list]:
    """The coordinate values of the dimensions as lists of plain python objects."""
    return [template[dim].values.tolist() for dim in dims]


def _nodes_or_none(values: np.ndarray) -> np.ndarray:
    """Object array with None instead of NaN, as a new array."""
    return np.where(pd.isnull(values), None, values)


def _unique(steps: typing.Iterable[ProcessingStep]) -> tuple:
    """The steps without duplicates, keeping the order."""
    return tuple({id(step): step for step in steps}.values())


def _create_nodes(
    nodes: np.ndarray,
    changed: np.ndarray,
    parents: list[np.ndarray],
    *,
    describe: typing.Callable[[dict[str, typing.Any]], ProcessingStepDescription],
    dims: list[typing.Hashable],
    coord_lists: list[list],
    shape: tuple[int, ...],
    times: np.ndarray,
    replaces: bool,
) -> None:
    """Set a new node with a new description per timeseries for the changed data points.

    All arrays have the layout (timeseries, time). ``nodes`` is modified in place, its old
    nodes are parents of the new nodes unless ``replaces``.
    """
    time_points = list(times)
    names = [str(dim) for dim in dims]
    for ts in np.flatnonzero(changed.any(axis=1)):
        index = np.unravel_index(ts, shape)
        coords = {n: values[i] for n, values, i in zip(names, coord_lists, index, strict=True)}
        description = describe(coords)
        columns = np.flatnonzero(changed[ts])
        old = [None] * len(columns) if replaces else nodes[ts, columns].tolist()
        extra = [p[ts, columns].tolist() for p in parents]
        new_nodes = []
        for k, t in enumerate(columns.tolist()):
            step_parents = [old[k]] if old[k] is not None else []
            for values in extra:
                value = values[k]
                if value is None:
                    continue
                if type(value) is tuple:
                    step_parents.extend(value)
                else:
                    step_parents.append(value)
            if len(step_parents) > 1:
                step_parents = _unique(step_parents)
            new_nodes.append(ProcessingStep(description, time_points[t], step_parents))
        nodes[ts, columns] = new_nodes


def _layout(da: xr.DataArray) -> tuple[list[typing.Hashable], xr.DataArray]:
    """The dimensions apart from "time", and the array with "time" as last dimension."""
    if "time" not in da.dims:
        raise ValueError(
            f"{da.name!r} has no 'time' dimension, its dimensions are {_dims_repr(da.dims)}."
        )
    dims = [dim for dim in da.dims if dim != "time"]
    return dims, da.transpose(*dims, "time")


def add_processing_step(
    processing_infos: xr.DataArray, step: ProcessingStepDescription, *, variable: str
) -> xr.DataArray:
    """Append a processing step to every data point of the input.

    Data points without processing information are left untouched.

    Parameters
    ----------
    processing_infos
        The processing information variable to add the step to. It is not modified.
    step
        The processing step to append. Its id is replaced by a new one for each
        timeseries, its variable by ``variable`` and its coordinates by those of each
        timeseries. The last step of each data point becomes the parent of the new step.
    variable
        The name of the data variable after the processing step.

    Returns
    -------
    with_step : xr.DataArray
        A copy of ``processing_infos`` with ``step`` appended to each data point.
    """
    dims, template = _layout(processing_infos)
    shape = template.shape
    nodes = _nodes_or_none(template.values).reshape(-1, shape[-1])
    result = nodes.copy()
    _create_nodes(
        result,
        ~pd.isnull(nodes),
        [],
        describe=lambda coords: evolve(step, id=uuid.uuid4(), variable=variable, coords=coords),
        dims=dims,
        coord_lists=_coordinate_lists(template, dims),
        shape=shape[:-1],
        times=template["time"].values,
        replaces=False,
    )
    return _from_layout(processing_infos, result.reshape(shape), dims)


def _from_layout(
    like: xr.DataArray, values: np.ndarray, dims: list[typing.Hashable]
) -> xr.DataArray:
    """The array with ``like``'s dimensions from values with the layout of _layout."""
    layout = [*dims, "time"]
    result = like.copy(deep=False)
    result.data = np.ascontiguousarray(values.transpose([layout.index(d) for d in like.dims]))
    return result


def _new_processing_infos(data: xr.DataArray, values: np.ndarray | None = None) -> xr.DataArray:
    """Processing information for ``data`` with the given values, by default empty."""
    if values is None:
        values = np.full(data.shape, None, dtype=object)
    name = processing_variable_name(data.name)
    return xr.DataArray(
        values,
        dims=data.dims,
        coords=data.coords,
        name=name,
        attrs={"entity": name, "described_variable": data.name},
    )


def _changed_values(old_da: xr.DataArray, new_da: xr.DataArray) -> xr.DataArray:
    """Boolean array which is True wherever ``new_da`` differs from ``old_da``.

    Values which are missing in both arrays count as unchanged, values which are
    missing in only one of them count as changed.
    """
    return (old_da != new_da) & ~(old_da.isnull() & new_da.isnull())


def _index_coordinates(da: xr.DataArray, index: tuple[int, ...]) -> dict[str, typing.Any]:
    """The coordinates of the value of ``da`` at ``index`` as plain python objects."""
    return {str(dim): da[dim].values[i].tolist() for dim, i in zip(da.dims, index, strict=True)}


def _coordinates_repr(coordinates: dict[str, typing.Any]) -> str:
    """Reduce the coordinates of a single timeseries to a short string representation."""
    return ", ".join(f"{dim}={value!r}" for dim, value in coordinates.items())


def _dims_repr(dims: typing.Iterable[typing.Hashable]) -> str:
    return repr(sorted(str(dim) for dim in dims))


def _described(da: xr.DataArray) -> xr.DataArray:
    """The array without its "time" dimension, i.e. with one value per timeseries."""
    if "time" not in da.dims:
        raise ValueError(
            f"{da.name!r} has no 'time' dimension, its dimensions are {_dims_repr(da.dims)}."
        )
    return da.isel(time=0, drop=True)


def _on_template(da: xr.DataArray, template: xr.DataArray, fill_value) -> np.ndarray:
    """The values of ``da`` for each value of ``template``, broadcasting ``da`` along
    dimensions it doesn't have."""
    if not set(da.dims) <= set(template.dims):
        raise ValueError(
            f"Dimensions of {da.name!r} {_dims_repr(da.dims)} are not a subset of the "
            f"dimensions of the result {_dims_repr(template.dims)}."
        )
    # non-index coordinates like a scalar coordinate of a selected value would conflict
    # with the dimensions of the result
    reindexed = da.reset_coords(drop=True).reindex(
        {dim: template[dim] for dim in da.dims}, fill_value=fill_value, copy=False
    )
    return reindexed.broadcast_like(template).transpose(*template.dims).values


def _exists(da: xr.DataArray, template: xr.DataArray) -> np.ndarray:
    """Boolean array which is True for each timeseries of ``template`` also in ``da``."""
    described = _described(da)
    present = xr.DataArray(
        np.ones(described.shape, dtype=bool),
        dims=described.dims,
        coords={dim: described[dim] for dim in described.dims},
        name=da.name,
    )
    return _on_template(present, template, fill_value=False)


def _nodes_on_template(da: xr.DataArray, template: xr.DataArray) -> np.ndarray:
    """The nodes in ``da`` for each data point of ``template``, None where there are none."""
    return _nodes_or_none(_on_template(da, template, fill_value=None))


def _processing_infos_on_template(
    data: xr.DataArray, processing_infos: xr.DataArray, template: xr.DataArray
) -> np.ndarray:
    """The processing information for each data point of ``template``."""
    try:
        xr.align(data, processing_infos, join="exact", copy=False)
    except ValueError as err:
        raise ValueError(
            f"Processing information {processing_infos.name!r} doesn't match the data it "
            f"describes: {err}"
        ) from err
    return _nodes_on_template(processing_infos, template)


def combine_histories(
    processing_infos: xr.DataArray,
    dim: typing.Hashable | list[typing.Hashable],
    *,
    data: xr.DataArray | None = None,
) -> xr.DataArray:
    """Combine the histories of all data points along ``dim``.

    Parameters
    ----------
    processing_infos
        The processing information to combine.
    dim
        The dimension or dimensions along which to combine.
    data
        Optional: the data described by ``processing_infos``. If given, the histories of
        data points without data are left out.

    Returns
    -------
    : xr.DataArray
        For each data point remaining after removing ``dim``, a tuple of the steps
        along ``dim``, for use as ``parents`` in record_step. Missing steps are left
        out.
    """
    dims = dim if isinstance(dim, list) else [dim]
    remaining = [x for x in processing_infos.dims if x not in dims]
    values = _nodes_or_none(processing_infos.transpose(*remaining, *dims).values)
    if data is not None:
        has_data = (
            data.notnull()
            .reindex_like(processing_infos, fill_value=False)
            .broadcast_like(processing_infos)
            .transpose(*remaining, *dims)
            .values
        )
        values = np.where(has_data, values, None)
    shape = values.shape[: len(remaining)]
    rows = values.reshape(-1, int(np.prod(values.shape[len(remaining) :], dtype=int))).tolist()
    result = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        result[i] = _unique(x for x in row if x is not None)
    return xr.DataArray(
        result.reshape(shape),
        dims=remaining,
        coords={x: processing_infos[x] for x in remaining},
        name=processing_infos.name,
    )


def record_step(
    new_da: xr.DataArray,
    *,
    previous: typing.Sequence[tuple[xr.DataArray, xr.DataArray]] = (),
    parents: typing.Sequence[xr.DataArray] = (),
    replaces: bool = False,
    function: str,
    description: str,
    source: str | None = None,
) -> xr.DataArray:
    """Processing information of ``new_da`` after a processing step.

    The previous state of each timeseries of ``new_da`` is taken from the first entry of
    ``previous`` which contains it. The processing information of a data point is:

    * the previous step, if the value is unchanged.
    * None, if the value was changed to a missing value.
    * a new step otherwise. Its parents are the previous step, unless ``replaces``, and
      the values of ``parents`` for the data point.

    All new steps of a timeseries share one description. Timeseries which did not exist
    before and contain no data have no processing information. Previous timeseries
    without data and without processing information count as not existing.

    Parameters
    ----------
    new_da
        The data after the processing step.
    previous
        Pairs of data before the processing step and its processing information. The data
        may lack dimensions of ``new_da``, it is broadcast along them.
    parents
        Additional parents of the new steps. Each value is a step, None, or a tuple of
        steps like returned by combine_histories, which can be steps of other time
        points. They may lack dimensions (also "time") or coordinates of ``new_da``.
    replaces
        If True, the changed values replace the previous values instead of being
        computed from them, so the previous steps are not parents of the new steps.
    function
        The name of the function which did the processing.
    description
        Human-readable description of the processing step.
    source
        Optional: a short identifier for the source of the data which was used for the
        processing.

    Returns
    -------
    : xr.DataArray
        The processing information of ``new_da``. The new descriptions record the name of
        ``new_da`` as their variable and the coordinates of their timeseries.

    Raises
    ------
    ValueError
        If processing information doesn't match the data it describes.
    """
    dims, template = _layout(new_da)
    ts_template = _described(template)
    shape = template.shape
    n_times = shape[-1]

    found = np.zeros(ts_template.shape, dtype=bool)
    nodes = np.full(shape, None, dtype=object)
    has_value = template.notnull().values
    # without a previous state, all data points with data are changed
    changed = has_value.copy()
    for data, processing_infos in previous:
        previous_nodes = _processing_infos_on_template(data, processing_infos, template)
        has_data = _on_template(data.notnull().any("time"), ts_template, fill_value=False)
        has_node = (~pd.isnull(previous_nodes)).any(axis=-1)
        # timeseries without data and processing information, e.g. added as padding when
        # aligning, carry no information
        exists = _exists(data, ts_template) & ~found & (has_data | has_node)
        nodes[exists] = previous_nodes[exists]
        reindexed = data.reindex({x: new_da[x] for x in data.dims}, copy=False)
        changed[exists] = (
            _changed_values(reindexed, template).transpose(*dims, "time").values[exists]
        )
        found |= exists
    changed_missing = changed & ~has_value
    nodes[changed_missing] = None
    changed &= has_value

    flat_nodes = nodes.reshape(-1, n_times)
    _create_nodes(
        flat_nodes,
        changed.reshape(-1, n_times),
        [_nodes_on_template(x, template).reshape(-1, n_times) for x in parents],
        describe=lambda coords: ProcessingStepDescription(
            function=function,
            description=description,
            source=source,
            variable=new_da.name,
            coords=coords,
        ),
        dims=dims,
        coord_lists=_coordinate_lists(template, dims),
        shape=shape[:-1],
        times=template["time"].values,
        replaces=replaces,
    )
    layout = [*dims, "time"]
    return _new_processing_infos(
        new_da,
        np.ascontiguousarray(nodes.transpose([layout.index(d) for d in new_da.dims])),
    )


# Storage format of processing information in netCDF files, see encode_processing_info.
PROCESSING_INFO_FORMAT = 2
FORMAT_ATTR = "processing_info_format"
_STEP = "processing step "
_NODE = "processing node "
_PARENT = "processing parent "


def _json_default(obj: typing.Any) -> typing.Any:
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Can not store {obj!r} as a coordinate of processing information.")


def _is_table_variable(name: typing.Hashable) -> bool:
    return isinstance(name, str) and name.startswith((_STEP, _NODE, _PARENT))


def _collect_nodes(
    values: typing.Iterable[np.ndarray],
) -> tuple[list[ProcessingStep], array.array, array.array]:
    """All nodes reachable from the arrays of nodes, parents first, and the parent table."""
    index: dict[int, int] = {}
    nodes: list[ProcessingStep] = []
    counts = array.array("q")
    parents = array.array("q")
    for array_ in values:
        for root in array_.ravel().tolist():
            if root is None or id(root) in index:
                continue
            # depth-first search without recursion, see ProcessingStep.history
            stack = [(root, False)]
            while stack:
                node, expanded = stack.pop()
                if expanded:
                    index[id(node)] = len(nodes)
                    nodes.append(node)
                    counts.append(len(node.parents))
                    parents.extend(index[id(p)] for p in node.parents)
                    continue
                if id(node) in index:
                    continue
                stack.append((node, True))
                stack.extend((p, False) for p in node.parents if id(p) not in index)
    return nodes, counts, parents


def encode_processing_info(ds: xr.Dataset) -> xr.Dataset:
    """Replace the processing steps in the dataset by tables of steps for writing to a file.

    The processing variables hold the index of the step of each data point in the table
    of steps, or -1. Steps reachable from several data points, also of different
    variables, are only stored once, steps not reachable from any data point are not
    stored. The dataset is not modified.
    """
    names = [name for name in ds.data_vars if is_processing_variable(name)]
    if not names:
        return ds
    values = {name: _nodes_or_none(ds[name].values) for name in names}
    nodes, counts, parents = _collect_nodes(values.values())
    node_index = {id(node): i for i, node in enumerate(nodes)}

    descriptions: dict[int, ProcessingStepDescription] = {}
    for node in nodes:
        descriptions.setdefault(id(node.description), node.description)
    description_list = list(descriptions.values())
    description_index = {id(d): i for i, d in enumerate(description_list)}

    result = ds.copy()
    for name in names:
        flat = values[name].ravel().tolist()
        indices = np.fromiter(
            (-1 if x is None else node_index[id(x)] for x in flat),
            dtype=np.int64,
            count=len(flat),
        )
        result[name] = ds[name].copy(data=indices.reshape(ds[name].shape))

    def optional(attribute: str) -> list[str]:
        return [getattr(d, attribute) or "" for d in description_list]

    def null(attribute: str) -> np.ndarray:
        return np.array([getattr(d, attribute) is None for d in description_list], dtype=np.int8)

    step_dim = "processing step"
    node_dim = "processing node"
    parent_dim = "processing parent"
    tables = {
        _STEP + "id": (step_dim, [d.id.hex for d in description_list]),
        _STEP + "function": (step_dim, [d.function for d in description_list]),
        _STEP + "description": (step_dim, [d.description for d in description_list]),
        _STEP + "source": (step_dim, optional("source")),
        _STEP + "source is null": (step_dim, null("source")),
        _STEP + "variable": (step_dim, optional("variable")),
        _STEP + "variable is null": (step_dim, null("variable")),
        _STEP + "coords": (
            step_dim,
            [json.dumps(d.coords, default=_json_default) for d in description_list],
        ),
        _NODE + "step": (node_dim, [description_index[id(n.description)] for n in nodes]),
        _NODE + "time": (node_dim, np.array([n.time for n in nodes], dtype="datetime64[ns]")),
        _NODE + "parents": (node_dim, np.frombuffer(counts, dtype=np.int64)),
        _PARENT + "node": (parent_dim, np.frombuffer(parents, dtype=np.int64)),
    }
    for name, (dim, data) in tables.items():
        data = np.asarray(data)
        if data.dtype.kind == "U" and data.size == 0:
            data = data.astype("<U1")
        result[name] = xr.DataArray(data, dims=[dim])
    result.attrs[FORMAT_ATTR] = PROCESSING_INFO_FORMAT
    return result


def decode_processing_info(ds: xr.Dataset) -> xr.Dataset:
    """Rebuild processing steps from the tables written by encode_processing_info and
    remove the tables.

    Raises
    ------
    ValueError
        If the dataset has processing variables in an unsupported format.
    """
    names = [name for name in ds.data_vars if is_processing_variable(name)]
    version = ds.attrs.get(FORMAT_ATTR)
    if not names:
        return ds
    if version != PROCESSING_INFO_FORMAT or any(ds[name].dtype.kind != "i" for name in names):
        raise ValueError(
            f"The file contains processing information in an unsupported format (version "
            f"{version!r}, supported: {PROCESSING_INFO_FORMAT}). Files with processing "
            f"information written by earlier development versions can not be read."
        )

    def column(name: str) -> np.ndarray:
        return ds[_STEP + name].values

    ids, functions, texts = column("id"), column("function"), column("description")
    sources, variables, coords = column("source"), column("variable"), column("coords")
    source_null, variable_null = column("source is null"), column("variable is null")
    descriptions = [
        ProcessingStepDescription(
            id=uuid.UUID(ids[i]),
            function=str(functions[i]),
            description=str(texts[i]),
            source=None if source_null[i] else str(sources[i]),
            variable=None if variable_null[i] else str(variables[i]),
            coords=json.loads(coords[i]),
        )
        for i in range(len(ids))
    ]

    node_steps = ds[_NODE + "step"].values.tolist()
    node_times = list(ds[_NODE + "time"].values)
    counts = ds[_NODE + "parents"].values.tolist()
    parents = ds[_PARENT + "node"].values.tolist()
    nodes: list[ProcessingStep] = []
    position = 0
    for step, time, count in zip(node_steps, node_times, counts, strict=True):
        if count:
            node_parents = tuple(map(nodes.__getitem__, parents[position : position + count]))
            position += count
        else:
            node_parents = ()
        nodes.append(ProcessingStep(descriptions[step], time, node_parents))

    node_array = np.empty(len(nodes) + 1, dtype=object)
    node_array[:-1] = nodes
    result = ds
    for name in names:
        # index -1 selects the last element, which is None
        restored = node_array[ds[name].values]
        result = result.assign({name: ds[name].copy(data=restored)})
    result = result.drop_vars([n for n in result.variables if _is_table_variable(n)])
    del result.attrs[FORMAT_ATTR]
    return result


class DataArrayProcessingInfoAccessor(BaseDataArrayAccessor):
    def with_creation_step(
        self,
        *,
        function: str,
        description: str,
        source: str | None = None,
    ) -> xr.Dataset:
        """Create a dataset with processing information from this DataArray.

        Every timeseries with data gets a processing step without parents which describes
        its creation. Use this to combine data without processing information with a
        dataset with processing information.

        Parameters
        ----------
        function
            The name of the function which created the data.
        description
            Human-readable description of how the data was created.
        source
            Optional: a short identifier for the source of the data.

        Returns
        -------
        : xr.Dataset
            A dataset containing this DataArray, named by its name, and its processing
            information.
        """
        if self._da.name is None:
            raise ValueError("The DataArray needs a name, which is used for the data variable.")
        return xr.Dataset(
            {
                self._da.name: self._da,
                processing_variable_name(self._da.name): record_step(
                    self._da, function=function, description=description, source=source
                ),
            }
        )


def _processing_infos_on_change(
    *,
    var: typing.Hashable,
    old_ds: xr.Dataset,
    new_da: xr.DataArray,
    other_ds: xr.Dataset | None,
    function: str,
    description: str,
    source: str | None,
) -> xr.DataArray:
    """The processing information of ``new_da``, see add_processing_step_on_change_ds."""
    name = processing_variable_name(var)
    previous = []
    parents = []
    if var in old_ds:
        old_da = old_ds[var]
        if set(old_da.dims) != set(new_da.dims):
            raise ValueError(
                f"Dimensions of {var!r} changed from {_dims_repr(old_da.dims)} to "
                f"{_dims_repr(new_da.dims)}."
            )
        previous.append((old_da, old_ds[name]))
    if other_ds is not None and name in other_ds:
        # timeseries which only exist in other_ds were taken from it, unless it lacks
        # dimensions: then they were broadcast to new coordinates, which is a change
        if set(other_ds[var].dims) == set(new_da.dims):
            previous.append((other_ds[var], other_ds[name]))
        parents.append(other_ds[name])
    return record_step(
        new_da,
        previous=previous,
        parents=parents,
        function=function,
        description=description,
        source=source,
    )


def _add_processing_step_on_change_ds(
    *,
    old_ds: xr.Dataset,
    new_ds: xr.Dataset,
    other_ds: xr.Dataset | None = None,
    function: str,
    description: str,
    source: str | None = None,
) -> xr.Dataset:
    """Determine the processing information of ``new_ds``, which was derived from
    ``old_ds`` and optionally ``other_ds``.

    For each timeseries of ``new_ds``, the processing information is:

    * the processing information of ``old_ds``, if the timeseries is unchanged compared
      to ``old_ds``.
    * the processing information of ``other_ds``, if the timeseries does not exist in
      ``old_ds``, but in ``other_ds``, and is unchanged compared to ``other_ds``. If
      ``other_ds`` lacks dimensions of the timeseries, it was broadcast and gets a new
      step.
    * a new processing step otherwise. Its parents are the processing information of
      ``old_ds`` and of ``other_ds`` at the same coordinates, if available. Its times
      are the changed time points.

    If ``old_ds`` has no processing information, the result has none either.

    Parameters
    ----------
    old_ds
        Pre-modification dataset including processing information.
    new_ds
        Post-modification dataset. Its data variables have to have the same dimensions
        as in ``old_ds``. Processing information contained in it is ignored.
    other_ds
        Optional: dataset including processing information from which the changed values
        in ``new_ds`` were taken. It may lack dimensions of ``new_ds``.
    function
        The name of the function which did the processing.
    description
        Human-readable description of the processing step.
    source
        Optional: a short identifier for the source of the data which was used for the
        processing.

    Returns
    -------
    : xr.Dataset
        A copy of ``new_ds`` with the processing information.

    Raises
    ------
    ValueError
        If the dimensions of a data variable changed, or if the processing information
        doesn't match the data it describes.
    """
    new_ds = _without_processing_info(new_ds)
    result = new_ds.copy()
    if not has_processing_info(old_ds):
        return result
    for var in new_ds.data_vars:
        result[processing_variable_name(var)] = _processing_infos_on_change(
            var=var,
            old_ds=old_ds,
            new_da=new_ds[var],
            other_ds=other_ds,
            function=function,
            description=description,
            source=source,
        )

    return result


def _without_processing_info(ds: xr.Dataset) -> xr.Dataset:
    return ds.drop_vars([var for var in ds if is_processing_variable(var)])


class ProcessingStepRecorder:
    """Records a processing step for every timeseries changed within a ``with`` block.

    Use it via :py:meth:`xarray.Dataset.pr.processing_step`. Within the block, ``ds`` is
    a deep copy of the dataset without processing information, and the result of the
    processing has to be assigned to ``ds``. After the block, ``result`` is the processed
    dataset including the updated processing information.

    How the processing information is determined is described in
    add_processing_step_on_change_ds.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        *,
        function: str,
        description: str,
        source: str | None = None,
        other_ds: xr.Dataset | None = None,
    ):
        ensure_complete_processing_info(ds)
        if other_ds is not None:
            ensure_complete_processing_info_of_other(ds, other_ds)
        self._original = ds
        self._other_ds = other_ds
        self._function = function
        self._description = description
        self._source = source
        self._result: xr.Dataset | None = None
        self.ds = _without_processing_info(ds).copy(deep=True)

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            return
        self._result = _add_processing_step_on_change_ds(
            old_ds=self._original,
            new_ds=self.ds,
            other_ds=self._other_ds,
            function=self._function,
            description=self._description,
            source=self._source,
        )
        if self.ds.equals(_without_processing_info(self._original)):
            logger.debug(
                f"No data changed in the processing step of {self._function!r}, so no "
                f"processing step was recorded."
            )

    @property
    def result(self) -> xr.Dataset:
        """The processed dataset including the updated processing information."""
        if self._result is None:
            raise RuntimeError(
                "The result is only available after the with block ended without an error."
            )
        return self._result


F = typing.TypeVar("F", bound=typing.Callable[..., xr.Dataset])


def records_processing_step(
    input_ds: str,
    *,
    description: str,
    function: str | None = None,
    source: str | None = None,
    other: str | None = None,
) -> typing.Callable[[F], F]:
    """Decorator which records a processing step for every timeseries changed by the
    decorated function.

    The decorator removes the processing information from a deep copy of the input
    dataset before the function is called, so the function only works with the data and
    returns the processed dataset. The decorator then adds the updated processing
    information to the returned dataset, determined like in
    :py:meth:`xarray.Dataset.pr.processing_step`.

    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> @records_processing_step("ds", description="interpolated with {method}")
    ... def interpolate(ds: xr.Dataset, method: str = "linear") -> xr.Dataset:
    ...     return ds.interpolate_na(dim="time", method=method)

    Parameters
    ----------
    input_ds
        The name of the argument holding the input dataset. For methods of primap2's
        dataset accessors, use "self": the method is then called on an accessor of the
        input dataset without processing information.
    description
        Human-readable description of the processing step. "{name}" fields are replaced
        by the arguments of the call using :py:meth:`str.format_map`.
    function
        Optional: the name of the function which does the processing. Defaults to the
        qualified name of the decorated function.
    source
        Optional: a short identifier for the source of the data which is used for the
        processing, with "{name}" fields replaced like in ``description``.
    other
        Optional: the name of the argument holding a dataset from which data is taken.
        The decorated function receives it without processing information. If the input
        dataset has processing information, a Dataset passed as ``other`` needs it as
        well and a DataArray is not allowed.
    """

    def decorator(func: F) -> F:
        signature = inspect.signature(func)
        for name in (input_ds, other):
            if name is not None and name not in signature.parameters:
                raise ValueError(f"{func.__qualname__} has no argument {name!r}.")
        function_name = func.__qualname__ if function is None else function

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> xr.Dataset:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)

            value = arguments[input_ds]
            is_accessor = isinstance(value, BaseDatasetAccessor)
            ds = value._ds if is_accessor else value
            if not isinstance(ds, xr.Dataset):
                raise TypeError(
                    f"Argument {input_ds!r} of {func.__qualname__} has to be a Dataset, not "
                    f"{type(ds).__name__}."
                )

            other_ds = None
            if other is not None:
                other_value = arguments[other]
                if isinstance(other_value, xr.Dataset | xr.DataArray):
                    ensure_complete_processing_info_of_other(ds, other_value)
                if isinstance(other_value, xr.Dataset):
                    bound.arguments[other] = _without_processing_info(other_value)
                    other_ds = other_value

            with ProcessingStepRecorder(
                ds,
                function=function_name,
                description=description.format_map(arguments),
                source=None if source is None else source.format_map(arguments),
                other_ds=other_ds,
            ) as step:
                bound.arguments[input_ds] = type(value)(step.ds) if is_accessor else step.ds
                result = func(*bound.args, **bound.kwargs)
                if not isinstance(result, xr.Dataset):
                    raise TypeError(
                        f"{func.__qualname__} has to return a Dataset, not {type(result).__name__}."
                    )
                step.ds = result
            return step.result

        return typing.cast(F, wrapper)

    return decorator


def used_sources(ds: xr.Dataset, dim: str = "used source") -> xr.Dataset:
    """Which sources were used for each data point, determined from the processing
    information. See :py:meth:`xarray.Dataset.pr.used_sources`."""
    raise NotImplementedError("being reworked")
