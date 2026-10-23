"""Processing information for the timeseries of a data variable."""

import functools
import inspect
import typing

import msgpack
import numpy as np
import pandas as pd
import xarray as xr
from attr import define, evolve, field
from loguru import logger

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor
from ._time_range import TimeRange, time_ranges

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
    """Raise if the dataset has processing information, but not for every timeseries.

    If a dataset has processing information, every data variable needs processing
    information, which may only be missing for timeseries without any data.

    Raises
    ------
    ValueError
        If processing information is missing for a data variable or for a timeseries
        with data.
    """
    if not has_processing_info(ds):
        return
    for var in ds.data_vars:
        if is_processing_variable(var):
            continue
        name = processing_variable_name(var)
        if name not in ds:
            raise ValueError(f"The dataset has processing information, but not for {var!r}.")
        processing_infos = ds[name]
        da = ds[var]
        has_data = da.notnull().any("time") if "time" in da.dims else da.notnull()
        missing = (has_data & processing_infos.isnull()).transpose(*processing_infos.dims)
        if missing.any():
            index = tuple(np.argwhere(missing.values)[0])
            coordinates = _coordinates_repr(_index_coordinates(processing_infos, index))
            raise ValueError(
                f"{int(missing.sum())} timeseries of {var!r} contain data, but have no "
                f"processing information, e.g. {coordinates}."
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


def _to_time_ranges(value: typing.Any) -> tuple[TimeRange, ...]:
    """Convert a TimeRange, an iterable of TimeRanges or of time points into TimeRanges."""
    if isinstance(value, TimeRange):
        return (value,)
    if isinstance(value, str):
        raise TypeError(f"time has to be a TimeRange or an iterable of time points, not {value!r}.")
    if not isinstance(value, np.ndarray):
        value = list(value)
        if all(isinstance(x, TimeRange) for x in value):
            return tuple(value)
    return time_ranges(value)


@define(frozen=True, kw_only=True)
class ProcessingStepDescription:
    """Structured description of a processing step done on a timeseries.

    Processing steps form the history of a timeseries: every step refers to the steps
    which produced its input data as its parents.

    Attributes
    ----------
    time
        Time points for which data was changed during the processing step. Can be given
        as a TimeRange, an iterable of TimeRanges or an iterable of time points, which
        is converted to TimeRanges.
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
    parents
        The last processing steps of the timeseries which were the input of this
        processing step. Empty if the step created the timeseries.
    """

    time: tuple[TimeRange, ...] = field(converter=_to_time_ranges)
    function: str
    description: str
    source: str | None = None
    variable: str | None = None
    coords: dict[str, typing.Any] | None = field(
        default=None, converter=lambda x: None if x is None else dict(x), hash=False
    )
    parents: tuple["ProcessingStepDescription", ...] = field(default=(), converter=tuple)

    def __str__(self) -> str:
        times = ", ".join(str(time_range) for time_range in self.time) or "none"
        text = self.function
        if self.source is not None:
            text += f" (data source: {self.source})"
        text += f" for {times}: {self.description}"
        timeseries = " ".join(
            x
            for x in (
                self.variable,
                None if not self.coords else _coordinates_repr(self.coords),
            )
            if x is not None
        )
        return f"{timeseries}: {text}" if timeseries else text

    def time_points(self) -> np.ndarray:
        """All time points for which data was changed during the processing step."""
        if not self.time:
            return np.array([], dtype="datetime64")
        return np.concatenate([time_range.time_points() for time_range in self.time])

    def history(self) -> list["ProcessingStepDescription"]:
        """All processing steps which led to this step, including the step itself."""
        ordered: list[ProcessingStepDescription] = []
        visited: set[int] = set()
        # depth-first search without recursion, so that long histories don't exhaust
        # the stack. Steps are marked as "expanded" once their parents are on the stack.
        stack: list[tuple[ProcessingStepDescription, bool]] = [(self, False)]
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
        stack: list[tuple[ProcessingStepDescription, str, str, str]] = [(self, "", "", "")]
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

    def unstructure(self) -> dict[str, typing.Any]:
        """Convert this step without its parents into basic python types."""
        return {
            "time": [time_range.unstructure() for time_range in self.time],
            "description": self.description,
            "function": self.function,
            "source": self.source,
            "variable": self.variable,
            "coords": self.coords,
        }

    @classmethod
    def structure(
        cls,
        u: dict[str, typing.Any],
        parents: typing.Iterable["ProcessingStepDescription"] = (),
    ) -> "ProcessingStepDescription":
        """Initialize from basic python types as created by "unstructure".

        Parameters
        ----------
        u
            The step as created by "unstructure".
        parents
            The parents of the step, which are not part of ``u``.
        """
        return cls(
            time=[TimeRange.structure(time_range) for time_range in u["time"]],
            function=u["function"],
            description=u["description"],
            source=u["source"],
            variable=u["variable"],
            coords=u["coords"],
            parents=parents,
        )

    def serialize(self) -> bytes:
        """Convert this step and all steps which led to it into binary data, e.g. for
        saving to disk.
        """
        history = self.history()
        positions = {id(step): i for i, step in enumerate(history)}
        # The parents have to come first: the binary data is saved as fixed-length byte
        # strings, which lose trailing null bytes, and the parent position 0 is encoded
        # as a null byte.
        return msgpack.packb(
            {
                "steps": [
                    {
                        "parents": [positions[id(parent)] for parent in step.parents],
                        **step.unstructure(),
                    }
                    for step in history
                ]
            },
            use_bin_type=True,
        )

    @staticmethod
    def serialize_optional(processing: "ProcessingStepDescription | None") -> bytes:
        """Convert into binary data, also for missing processing information.

        Processing information can be missing for individual timeseries, for example
        if a dataset uses different categories for different variables. Missing
        processing information is represented by empty binary data.

        Parameters
        ----------
        processing
            The last processing step of a timeseries, or a null value (``None`` or NaN)
            if no processing information is available for the timeseries.
        """
        if pd.isnull(processing):
            return b""
        return processing.serialize()

    @classmethod
    def deserialize(cls, b: bytes) -> "ProcessingStepDescription | None":
        """Parse from binary data as produced by "serialize" or "serialize_optional".

        Parameters
        ----------
        b
            Binary data representing the processing steps of a timeseries, or
            empty binary data if no processing information is available for the
            timeseries.

        Returns
        -------
        processing : ProcessingStepDescription or None
            The last processing step of the timeseries, which refers to the earlier
            steps as its parents. ``None`` is returned for empty binary data.
        """
        if not b:
            return None
        ust = msgpack.unpackb(b, raw=False, use_list=False)
        steps: list[ProcessingStepDescription] = []
        for u in ust["steps"]:
            steps.append(cls.structure(u, parents=[steps[i] for i in u["parents"]]))
        return steps[-1]


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


def add_processing_step(
    processing_infos: xr.DataArray, step: ProcessingStepDescription, *, variable: str
) -> xr.DataArray:
    """Append a processing step to every timeseries of the input.

    Timeseries whose processing information is missing are left untouched.

    Parameters
    ----------
    processing_infos
        The processing information variable to add the step to. It is not modified.
    step
        The processing step to append. Its parents are replaced by the last processing
        step of each timeseries, its variable by ``variable`` and its coordinates by
        those of each timeseries.
    variable
        The name of the data variable after the processing step.

    Returns
    -------
    with_step : xr.DataArray
        A copy of ``da`` with ``step`` appended to each timeseries.
    """
    if len(step.parents) > 0:
        raise ValueError(
            "Calling add_processing_step with a history of steps instead  of an individual step "
            "is invalid. (Ensure the passed step has no parents!)"
        )

    values = processing_infos.values.copy()
    for index in np.ndindex(values.shape):
        if pd.isnull(values[index]):
            continue
        values[index] = evolve(
            step,
            variable=variable,
            coords=_index_coordinates(processing_infos, index),
            parents=(values[index],),
        )
    result = processing_infos.copy(deep=False)
    result.data = values
    return result


def _new_processing_infos(data: xr.DataArray, values: np.ndarray | None = None) -> xr.DataArray:
    """Processing information for ``data`` with the given values, by default empty."""
    template = _described(data)
    if values is None:
        values = np.full(template.shape, None, dtype=object)
    name = processing_variable_name(data.name)
    return xr.DataArray(
        values,
        dims=template.dims,
        coords=template.coords,
        name=name,
        attrs={"entity": name, "described_variable": data.name},
    )


def _changed_values(old_da: xr.DataArray, new_da: xr.DataArray) -> xr.DataArray:
    """Boolean array which is True wherever ``new_da`` differs from ``old_da``.

    Values which are missing in both arrays count as unchanged, values which are
    missing in only one of them count as changed.
    """
    return (old_da != new_da) & ~(old_da.isnull() & new_da.isnull())


def _coordinate_value(coords: xr.DataArray, index: int) -> typing.Any:
    """A single coordinate value as a plain python object."""
    value = coords.data[index]
    return value.item() if isinstance(value, np.generic) else value


def _index_coordinates(da: xr.DataArray, index: tuple[int, ...]) -> dict[str, typing.Any]:
    """The coordinates of the value of ``da`` at ``index`` as plain python objects."""
    return {str(dim): _coordinate_value(da[dim], i) for dim, i in zip(da.dims, index, strict=True)}


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
    """The values of ``da`` for each timeseries of ``template``, broadcasting ``da`` along
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


def _processing_infos_on_template(
    data: xr.DataArray, processing_infos: xr.DataArray, template: xr.DataArray
) -> np.ndarray:
    """The processing information for each timeseries of ``template``."""
    try:
        xr.align(_described(data), processing_infos, join="exact")
    except ValueError as err:
        raise ValueError(
            f"Processing information {processing_infos.name!r} doesn't match the data it "
            f"describes: {err}"
        ) from err
    return _on_template(processing_infos, template, fill_value=None)


def combine_histories(
    processing_infos: xr.DataArray,
    dim: typing.Hashable | list[typing.Hashable],
    *,
    data: xr.DataArray | None = None,
) -> xr.DataArray:
    """Combine the histories of all timeseries along ``dim``.

    Parameters
    ----------
    processing_infos
        The processing information to combine.
    dim
        The dimension or dimensions along which to combine.
    data
        Optional: the data described by ``processing_infos``. If given, the histories of
        timeseries without any data are left out.

    Returns
    -------
    : xr.DataArray
        For each timeseries remaining after removing ``dim``, a tuple of the histories
        along ``dim``, for use as ``parents`` in record_step. Missing histories are left
        out.
    """
    dims = dim if isinstance(dim, list) else [dim]
    if data is not None:
        has_data = data.notnull().any("time").reindex_like(processing_infos, fill_value=False)
        processing_infos = processing_infos.where(has_data)
    remaining = [x for x in processing_infos.dims if x not in dims]
    values = processing_infos.transpose(*remaining, *dims).values
    shape = values.shape[: len(remaining)]
    values = values.reshape(*shape, -1)
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        result[index] = _unique(x for x in values[index] if not pd.isnull(x))
    return xr.DataArray(
        result,
        dims=remaining,
        coords={x: processing_infos[x] for x in remaining},
        name=processing_infos.name,
    )


def _unique(steps: typing.Iterable[ProcessingStepDescription]) -> tuple:
    """The steps without duplicates, keeping the order."""
    return tuple({id(step): step for step in steps}.values())


def record_step(
    new_da: xr.DataArray,
    *,
    previous: typing.Sequence[tuple[xr.DataArray, xr.DataArray]] = (),
    parents: typing.Sequence[xr.DataArray] = (),
    function: str,
    description: str,
    source: str | None = None,
) -> xr.DataArray:
    """Processing information of ``new_da`` after a processing step.

    The previous state of each timeseries of ``new_da`` is taken from the first entry of
    ``previous`` which contains it. The processing information of the timeseries is:

    * the previous processing information, if the timeseries is unchanged.
    * a new processing step, if the timeseries was changed or did not exist before. Its
      parents are the previous processing information, if available, and the values of
      ``parents`` for the timeseries. Its times are the changed time points.

    Timeseries which did not exist before and contain no data have no processing
    information. Previous timeseries without data and without processing information
    count as not existing.

    Parameters
    ----------
    new_da
        The data after the processing step.
    previous
        Pairs of data before the processing step and its processing information. The data
        may lack dimensions of ``new_da``, it is broadcast along them.
    parents
        Additional parents of the new steps, as processing information or as tuples of
        processing information like returned by combine_histories. They may lack
        dimensions or timeseries of ``new_da``.
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
        The processing information of ``new_da``. The new steps record the name of
        ``new_da`` as their variable and the coordinates of their timeseries.

    Raises
    ------
    ValueError
        If processing information doesn't match the data it describes.
    """
    template = _described(new_da)
    dims = template.dims
    shape = template.shape

    found = np.zeros(shape, dtype=bool)
    result = np.full(shape, None, dtype=object)
    # without a previous state, all time points with data are changed
    changed = new_da.notnull().transpose(*dims, "time").values
    for data, processing_infos in previous:
        previous_infos = _processing_infos_on_template(data, processing_infos, template)
        has_data = _on_template(data.notnull().any("time"), template, fill_value=False)
        # timeseries without data and processing information, e.g. added as padding when
        # aligning, carry no information
        exists = _exists(data, template) & ~found & (has_data | ~pd.isnull(previous_infos))
        result[exists] = previous_infos[exists]
        reindexed = data.reindex({x: new_da[x] for x in data.dims}, copy=False)
        changed[exists] = _changed_values(reindexed, new_da).transpose(*dims, "time").values[exists]
        found |= exists
    result[pd.isnull(result)] = None
    parent_values = [_on_template(x, template, fill_value=None) for x in parents]

    times = new_da["time"].values
    for flat_index in np.flatnonzero(changed.any(axis=-1)):
        index = np.unravel_index(flat_index, shape)
        step_parents = [] if result[index] is None else [result[index]]
        for value in parent_values:
            if isinstance(value[index], tuple):
                step_parents.extend(value[index])
            elif not pd.isnull(value[index]):
                step_parents.append(value[index])
        result[index] = ProcessingStepDescription(
            time=times[changed[index]],
            function=function,
            description=description,
            source=source,
            variable=new_da.name,
            coords=_index_coordinates(template, index),
            parents=_unique(step_parents),
        )

    return _new_processing_infos(new_da, result)


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


# levels of source usage in used_sources, chosen so that combining is taking the maximum
_USED_AT_OTHER_TIMES = 1
_USED_AT_SAME_TIME = 2

# per step and label: bit masks over the time axis of the data used at the same time
# point and of the data used only at other time points
_Usage = dict[typing.Hashable, tuple[int, int]]


def _is_previous_state(step: ProcessingStepDescription, parent: ProcessingStepDescription) -> bool:
    """True if ``parent`` is the state of the timeseries of ``step`` before the step."""
    if step.coords is None or parent.coords is None:
        # without coordinates, rely on the convention to list the previous state first
        return True
    return parent.variable == step.variable and all(
        dim in step.coords and step.coords[dim] == value for dim, value in parent.coords.items()
    )


def _step_usage(
    step: ProcessingStepDescription, parent_usages: list[_Usage], changed: int
) -> _Usage:
    if not step.parents:
        if step.source is None:
            raise ValueError(f"The processing step {step} created data, but has no source.")
        return {step.source: (changed, 0)} if changed else {}
    usage: _Usage = {}

    def add(key: typing.Hashable, same: int, other: int) -> None:
        old_same, old_other = usage.get(key, (0, 0))
        usage[key] = (old_same | same, old_other | other)

    if _is_previous_state(step, step.parents[0]):
        for key, (same, other) in parent_usages[0].items():
            if (same | other) & ~changed:
                add(key, same & ~changed, other & ~changed)
    covered = 0
    for parent_usage in parent_usages:
        for key, (same, other) in parent_usage.items():
            if (same | other) & changed:
                add(key, same & changed, other & changed)
                covered |= (same | other) & changed
    # values not derived from any parent at the same time point come from the step's own
    # source, or were derived from other time points, e.g. by extrapolation
    uncovered = changed & ~covered
    if uncovered and step.source is not None:
        add(step.source, uncovered, 0)
    elif uncovered:
        for parent_usage in parent_usages:
            for key, (same, other) in parent_usage.items():
                if same | other:
                    add(key, 0, uncovered)
    return usage


def _bits_to_array(bits: int, length: int) -> np.ndarray:
    as_bytes = np.frombuffer(bits.to_bytes((length + 7) // 8, "little"), dtype=np.uint8)
    return np.unpackbits(as_bytes, bitorder="little")[:length].astype(bool)


def used_sources(ds: xr.Dataset, dim: str = "used source") -> xr.Dataset:
    """Which sources were used for each data point, determined from the processing
    information. See :py:meth:`xarray.Dataset.pr.used_sources`."""
    if not has_processing_info(ds):
        raise ValueError("The dataset has no processing information.")
    ensure_complete_processing_info(ds)

    times = ds["time"].values
    position = {t: i for i, t in enumerate(times.astype(np.int64))}
    changed_cache: dict[tuple[TimeRange, ...], int] = {}

    def changed_bits(step: ProcessingStepDescription) -> int:
        if step.time not in changed_cache:
            points = step.time_points().astype(times.dtype).astype(np.int64)
            changed_cache[step.time] = sum(1 << position[t] for t in points if t in position)
        return changed_cache[step.time]

    usages: dict[int, _Usage] = {}

    def usage_of(last_step: ProcessingStepDescription) -> _Usage:
        # depth-first search without recursion, see ProcessingStepDescription.history
        stack = [(last_step, False)]
        while stack:
            step, expanded = stack.pop()
            if id(step) in usages:
                continue
            if expanded:
                usages[id(step)] = _step_usage(
                    step, [usages[id(p)] for p in step.parents], changed_bits(step)
                )
                continue
            stack.append((step, True))
            stack.extend((p, False) for p in step.parents if id(p) not in usages)
        return usages[id(last_step)]

    variables = [var for var in ds.data_vars if not is_processing_variable(var)]
    cell_usages = {}
    keys: set[typing.Hashable] = set()
    for var in variables:
        processing = ds[processing_variable_name(var)]
        cell_usages[var] = [
            None if step is None else usage_of(step) for step in processing.values.flat
        ]
        keys.update(key for usage in cell_usages[var] if usage for key in usage)
    keys_sorted = sorted(keys, key=str)
    key_index = {key: i for i, key in enumerate(keys_sorted)}

    result = xr.Dataset(attrs=ds.attrs)
    for var in variables:
        processing = ds[processing_variable_name(var)]
        levels = np.zeros((processing.size, len(times), len(keys_sorted)), dtype=np.int8)
        for cell, usage in enumerate(cell_usages[var]):
            for key, (same, other) in (usage or {}).items():
                column = levels[cell, :, key_index[key]]
                column[_bits_to_array(other, len(times))] = _USED_AT_OTHER_TIMES
                column[_bits_to_array(same, len(times))] = _USED_AT_SAME_TIME
        levels = levels.reshape((*processing.shape, len(times), len(keys_sorted)))
        # data points without data used no sources
        levels *= ds[var].notnull().transpose(*processing.dims, "time").values[..., np.newaxis]
        da = xr.DataArray(
            levels,
            dims=(*processing.dims, "time", dim),
            coords={**processing.coords, "time": ds["time"], dim: keys_sorted},
        )
        result[var] = da.transpose(*ds[var].dims, dim).assign_attrs(
            flag_values=[0, _USED_AT_OTHER_TIMES, _USED_AT_SAME_TIME],
            flag_meanings="not_used used_at_other_time_points used_at_this_time_point",
        )
    return result
