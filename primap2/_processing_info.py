"""Processing information for the timeseries of a data variable."""

import typing

import msgpack
import numpy as np
import pandas as pd
import xarray as xr
from attr import define, evolve, field
from loguru import logger

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
    if any(is_processing_variable(var) for var in ds):
        raise NotImplementedError(
            "Dataset contains processing information, this is not supported yet. "
            "Use ds.pr.remove_processing_info()."
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
    parents
        The last processing steps of the timeseries which were the input of this
        processing step. Empty if the step created the timeseries.
    """

    time: tuple[TimeRange, ...] = field(converter=_to_time_ranges)
    function: str
    description: str
    source: str | None = None
    parents: tuple["ProcessingStepDescription", ...] = field(default=(), converter=tuple)

    def __str__(self) -> str:
        times = ", ".join(str(time_range) for time_range in self.time) or "none"
        if self.source is None:
            return f"Using function={self.function} for times={times}: {self.description}"
        else:
            return (
                f"Using function={self.function} with source={self.source} for "
                f"times={times}: {self.description}"
            )

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
        """Human-readable description of all processing steps which led to this step."""
        history = self.history()
        numbers = {id(step): i for i, step in enumerate(history, start=1)}
        lines = []
        for step in history:
            line = f"[{numbers[id(step)]}] "
            if step.parents:
                parent_numbers = ", ".join(f"[{numbers[id(parent)]}]" for parent in step.parents)
                line += f"(from {parent_numbers}) "
            lines.append(line + str(step))
        return "\n".join(lines)

    def unstructure(self) -> dict[str, typing.Any]:
        """Convert this step without its parents into basic python types."""
        return {
            "time": [time_range.unstructure() for time_range in self.time],
            "description": self.description,
            "function": self.function,
            "source": self.source,
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


def add_processing_step(
    processing_infos: xr.DataArray, step: ProcessingStepDescription
) -> xr.DataArray:
    """Append a processing step to every timeseries of the input.

    Timeseries whose processing information is missing are left untouched.

    Parameters
    ----------
    processing_infos
        The processing information variable to add the step to. It is not modified.
    step
        The processing step to append. Its parents are replaced by the last processing
        step of each timeseries.

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

    def append(processing: ProcessingStepDescription | None):
        if pd.isnull(processing):
            return processing
        return evolve(step, parents=(processing,))

    result = processing_infos.copy(deep=False)
    result.data = np.vectorize(append, otypes=[object])(processing_infos.data)
    return result


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
    previous: typing.Sequence[tuple[xr.DataArray, xr.DataArray | None]] = (),
    parents: typing.Sequence[xr.DataArray] = (),
    function: str,
    description_template: str,
    source: str | None = None,
) -> xr.DataArray:
    """Processing information of ``new_da`` after a processing step.

    The previous state of each timeseries of ``new_da`` is taken from the first entry of
    ``previous`` which contains it. The processing information of the timeseries is:

    * the previous processing information, if the timeseries is unchanged.
    * a new processing step, if the timeseries was changed or did not exist before. Its
      parents are the previous processing information and the values of ``parents``
      for the timeseries. Its times are the changed time points.

    Missing previous processing information stays missing, and timeseries which did not
    exist before and contain no data have no processing information. Previous timeseries
    without data and without processing information count as not existing.

    Parameters
    ----------
    new_da
        The data after the processing step.
    previous
        Pairs of data before the processing step and its processing information, which
        is ``None`` if the data has no processing information. The data may lack
        dimensions of ``new_da``, it is broadcast along them.
    parents
        Additional parents of the new steps, as processing information or as tuples of
        processing information like returned by combine_histories. They may lack
        dimensions or timeseries of ``new_da``.
    function
        The name of the function which did the processing.
    description_template
        Human-readable description of the processing step, in which "<coords>" is
        replaced by the coordinates of the timeseries.
    source
        Optional: a short identifier for the source of the data which was used for the
        processing.

    Returns
    -------
    : xr.DataArray
        The processing information of ``new_da``.

    Raises
    ------
    ValueError
        If processing information doesn't match the data it describes, or if a timeseries
        contains data, did not exist before, and has no parents.
    """
    template = _described(new_da)
    dims = template.dims
    shape = template.shape

    found = np.zeros(shape, dtype=bool)
    result = np.full(shape, None, dtype=object)
    # without a previous state, all time points with data are changed
    changed = new_da.notnull().transpose(*dims, "time").values
    for data, processing_infos in previous:
        if processing_infos is None:
            previous_infos = np.full(shape, None, dtype=object)
        else:
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
        if found[index]:
            if result[index] is None:
                continue
            step_parents = [result[index]]
        else:
            step_parents = []
        for value in parent_values:
            if isinstance(value[index], tuple):
                step_parents.extend(value[index])
            elif not pd.isnull(value[index]):
                step_parents.append(value[index])
        coordinates = _coordinates_repr(
            {
                str(dim): _coordinate_value(template[dim], i)
                for dim, i in zip(dims, index, strict=True)
            }
        )
        if not step_parents:
            raise ValueError(
                f"The timeseries {coordinates} of {new_da.name!r} contains data, but did "
                f"not exist before and has no parents, so its history is unknown."
            )
        result[index] = ProcessingStepDescription(
            time=times[changed[index]],
            function=function,
            description=description_template.replace("<coords>", coordinates),
            source=source,
            parents=_unique(step_parents),
        )

    name = processing_variable_name(new_da.name)
    return xr.DataArray(
        result,
        dims=dims,
        coords=template.coords,
        name=name,
        attrs={"entity": name, "described_variable": new_da.name},
    )


def _processing_infos_on_change(
    *,
    var: typing.Hashable,
    old_ds: xr.Dataset,
    new_da: xr.DataArray,
    other_ds: xr.Dataset | None,
    function: str,
    description_template: str,
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
    if other_ds is not None and var in other_ds:
        # timeseries which only exist in other_ds were taken from it
        other_infos = other_ds.get(name)
        previous.append((other_ds[var], other_infos))
        if other_infos is not None:
            parents.append(other_infos)
    return record_step(
        new_da,
        previous=previous,
        parents=parents,
        function=function,
        description_template=description_template,
        source=source,
    )


def add_processing_step_on_change_ds(
    *,
    old_ds: xr.Dataset,
    new_ds: xr.Dataset,
    other_ds: xr.Dataset | None = None,
    function: str,
    description_template: str,
    source: str | None = None,
) -> xr.Dataset:
    """Determine the processing information of ``new_ds``, which was derived from
    ``old_ds`` and optionally ``other_ds``.

    For each timeseries of ``new_ds``, the processing information is:

    * the processing information of ``old_ds``, if the timeseries is unchanged compared
      to ``old_ds``.
    * a new processing step, if the timeseries was changed compared to ``old_ds``. Its
      parents are the processing information of ``old_ds`` and of ``other_ds`` at the
      same coordinates, if available. ``description_template`` is used as a template for
      its description, in which "<coords>" is replaced by the coordinates of the
      timeseries and "<var>" by the name of its data variable. Its times are the changed
      time points.
    * the processing information of ``other_ds``, if the timeseries does not exist in
      ``old_ds``, but in ``other_ds``.

    Processing information which is missing in ``old_ds`` stays missing, and variables
    without processing information stay without processing information.

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
    description_template
        Human-readable description of the processing step, optionally including the "<coords>"
        and "<var>" placeholders.
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
        If the dimensions of a data variable changed, if the processing information
        doesn't match the data it describes, or if a timeseries of ``new_ds`` contains
        data but exists neither in ``old_ds`` nor in ``other_ds``.
    """
    new_ds = _without_processing_info(new_ds)
    result = new_ds.copy()
    for var in new_ds.data_vars:
        name = processing_variable_name(var)
        if var in old_ds:
            tracked = name in old_ds
        elif other_ds is not None and var in other_ds:
            tracked = name in other_ds
        else:
            tracked = any(is_processing_variable(x) for x in old_ds) or (
                other_ds is not None and any(is_processing_variable(x) for x in other_ds)
            )
            if tracked:
                raise ValueError(
                    f"{var!r} exists neither in old_ds nor in other_ds, so its history is unknown."
                )
        if not tracked:
            continue
        result[name] = _processing_infos_on_change(
            var=var,
            old_ds=old_ds,
            new_da=new_ds[var],
            other_ds=other_ds,
            function=function,
            description_template=description_template.replace("<var>", str(var)),
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
        description_template: str,
        source: str | None = None,
        other_ds: xr.Dataset | None = None,
    ):
        self._original = ds
        self._other_ds = other_ds
        self._function = function
        self._description_template = description_template
        self._source = source
        self._result: xr.Dataset | None = None
        self.ds = _without_processing_info(ds).copy(deep=True)

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            return
        self._result = add_processing_step_on_change_ds(
            old_ds=self._original,
            new_ds=self.ds,
            other_ds=self._other_ds,
            function=self._function,
            description_template=self._description_template,
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
