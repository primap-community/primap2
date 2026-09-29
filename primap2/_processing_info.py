"""Processing information for the timeseries of a data variable."""

import typing

import msgpack
import numpy as np
import pandas as pd
import xarray as xr
from attr import define

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


@define(frozen=True, kw_only=True)
class ProcessingStepDescription:
    """Structured description of a processing step done on a timeseries.

    Attributes
    ----------
    time
        Time points for which data was changed during the processing step. Use
        "all" if all time points were changed or it is not specified which time
        points were changed.
    function
        The name of the function which did the processing.
    description
        Human-readable description of the processing step.
    source
        Optional: a short identifier for the source of the data which was used for the
        processing.
    """

    time: np.ndarray[np.datetime64] | typing.Literal["all"]
    function: str
    description: str
    source: str | None = None

    def __str__(self) -> str:
        if self.source is None:
            return f"Using function={self.function} for times={self.time}: {self.description}"
        else:
            return (
                f"Using function={self.function} with source={self.source} for "
                f"times={self.time}: {self.description}"
            )

    def unstructure(self) -> dict[str, typing.Any]:
        """Convert into basic python types."""
        return {
            "time": "all"
            if isinstance(self.time, str) and self.time == "all"
            else list(
                np.datetime_as_string(
                    self.time,
                    unit="Y",
                )
            ),
            "description": self.description,
            "function": self.function,
            "source": self.source,
        }

    @classmethod
    def structure(cls, u: dict[str, typing.Any]) -> "ProcessingStepDescription":
        """Initialize from basic python types as created by "unstructure"."""
        time = u.pop("time")
        return cls(time="all" if time == "all" else np.array(time, dtype=np.datetime64), **u)


@define(frozen=True)
class TimeseriesProcessingDescription:
    """Structured description of all processing steps done on a timeseries.

    Attributes
    ----------
    steps
        Steps that were performed during processing, in order from first to last.
    """

    steps: list[ProcessingStepDescription]

    def __str__(self) -> str:
        return "\n".join(str(step) for step in self.steps)

    def serialize(self) -> bytes:
        """Convert into binary data, e.g. for saving to disk."""
        return msgpack.packb({"steps": [x.unstructure() for x in self.steps]}, use_bin_type=True)

    @staticmethod
    def serialize_optional(
        processing: "TimeseriesProcessingDescription | None",
    ) -> bytes:
        """Convert into binary data, also for missing processing information.

        Processing information can be missing for individual timeseries, for example
        if a dataset uses different categories for different variables. Missing
        processing information is represented by empty binary data.

        Parameters
        ----------
        processing
            A TimeseriesProcessingDescription, or a null value (``None`` or NaN) if
            no processing information is available for the timeseries.
        """
        if pd.isnull(processing):
            return b""
        return processing.serialize()

    @classmethod
    def deserialize(cls, b: bytes) -> "TimeseriesProcessingDescription | None":
        """Parse from binary data as produced by "serialize" or "serialize_optional".

        Parameters
        ----------
        b
            Binary data representing a TimeseriesProcessingDescription, or empty
            binary data if no processing information is available for the timeseries.

        Returns
        -------
        processing : TimeseriesProcessingDescription or None
            ``None`` is returned for empty binary data.
        """
        if not b:
            return None
        ust = msgpack.unpackb(b, raw=False, use_list=False)
        return cls(steps=[ProcessingStepDescription.structure(x) for x in ust["steps"]])


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
        The processing step to append.

    Returns
    -------
    with_step : xr.DataArray
        A copy of ``da`` with ``step`` appended to each timeseries.
    """

    def append(processing: TimeseriesProcessingDescription | None):
        if pd.isnull(processing):
            return processing
        return TimeseriesProcessingDescription(steps=[*processing.steps, step])

    result = processing_infos.copy()
    result.data = np.vectorize(append, otypes=[object])(processing_infos.data)
    return result


def _changed_values(old_da: xr.DataArray, new_da: xr.DataArray) -> xr.DataArray:
    """Boolean array which is True wherever ``new_da`` differs from ``old_da``.

    Values which are missing in both arrays count as unchanged, values which are
    missing in only one of them count as changed.
    """
    return (old_da != new_da) & ~(old_da.isnull() & new_da.isnull())


def _ensure_matching_dimensions(
    *, old_da: xr.DataArray, new_da: xr.DataArray, processing_infos: xr.DataArray
) -> None:
    """Ensure that data and processing information describe the same timeseries.

    Raises
    ------
    ValueError
        If the dimensions or coordinates of the given arrays don't match.
    """
    if "time" not in new_da.dims:
        raise ValueError(
            f"The data has no 'time' dimension, its dimensions are "
            f"{sorted(str(dim) for dim in new_da.dims)!r}."
        )
    if set(old_da.dims) != set(new_da.dims):
        raise ValueError(
            f"Dimensions of old_da {sorted(str(dim) for dim in old_da.dims)!r} and new_da "
            f"{sorted(str(dim) for dim in new_da.dims)!r} don't match."
        )
    described_dims = {dim for dim in new_da.dims if dim != "time"}
    if described_dims != set(processing_infos.dims):
        raise ValueError(
            f"Dimensions of the processing information "
            f"{sorted(str(dim) for dim in processing_infos.dims)!r} don't match the "
            f"dimensions of the data {sorted(str(dim) for dim in described_dims)!r}."
        )

    # xarray would silently align the arrays on the intersection of their coordinates,
    # which would compare and describe the wrong timeseries, so require equal coordinates
    try:
        xr.align(old_da, new_da, processing_infos, join="exact")
    except ValueError as err:
        raise ValueError(f"Coordinate values of the given arrays don't match: {err}") from err


def _coordinate_value(coords: xr.DataArray, index: int) -> typing.Any:
    """A single coordinate value as a plain python object."""
    value = coords.data[index]
    return value.item() if isinstance(value, np.generic) else value


def _coordinates_repr(coordinates: dict[str, typing.Any]) -> str:
    """Reduce the coordinates of a single timeseries to a short string representation."""
    return ", ".join(f"{dim}={value!r}" for dim, value in coordinates.items())


def add_processing_step_on_change(
    *,
    old_da: xr.DataArray,
    new_da: xr.DataArray,
    processing_infos: xr.DataArray,
    function: str,
    description_template: str,
    source: str | None = None,
) -> xr.DataArray:
    """For every timeseries of ``new_da`` that was changed compared to ``old_da``, append a new
    processing step to the corresponding processing information. ``description_template`` is used
    as a template for the appended processing steps description in which "<coords>" will be replaced
    by the coordinates of the affected timeseries. The times will be set automatically to a list of
    the changed timepoints.

    Timeseries whose processing information is missing are left untouched.

    Parameters
    ----------
    old_da
        Pre-modification data.
    new_da
        Post-modification data. Has to have the same dimensions and coordinates as
        ``old_da``, only the values may differ.
    processing_infos
        The processing information variable describing ``old_da``. Has to have the same
        dimensions and coordinates as the data, with the exception of the "time"
        dimension, which it does not have. It is not modified.
    function
        The name of the function which did the processing.
    description_template
        Human-readable description of the processing step, optionally including the "<coords>"
        placeholder.
    source
        Optional: a short identifier for the source of the data which was used for the
        processing.

    Returns
    -------
    : xr.DataArray
        A copy of ``processing_infos`` with the new processing step appended to each
        affected timeseries.

    Raises
    ------
    ValueError
        If the dimensions or coordinates of the data and the processing information
        don't match.
    """
    _ensure_matching_dimensions(old_da=old_da, new_da=new_da, processing_infos=processing_infos)

    changed = _changed_values(old_da, new_da)
    times = changed["time"].data
    result = processing_infos.copy()
    for index in np.ndindex(processing_infos.shape):
        processing = processing_infos.data[index]
        if pd.isnull(processing):
            continue

        selection = dict(zip(processing_infos.dims, index, strict=True))
        changed_times = times[changed.isel(selection).data]
        if not len(changed_times):
            continue

        coordinates = {
            str(dim): _coordinate_value(processing_infos.coords[dim], i)
            for dim, i in selection.items()
            if dim in processing_infos.coords
        }

        step = ProcessingStepDescription(
            time=changed_times,
            function=function,
            description=description_template.replace("<coords>", _coordinates_repr(coordinates)),
            source=source,
        )
        result.data[index] = TimeseriesProcessingDescription(steps=[*processing.steps, step])

    return result


def add_processing_step_on_change_ds(
    *,
    old_ds: xr.Dataset,
    new_ds: xr.Dataset,
    function: str,
    description_template: str,
    source: str | None = None,
) -> xr.Dataset:
    """For every timeseries of ``new_ds`` that was changed compared to ``old_ds``, append a new
    processing step to the corresponding processing information. ``description_template`` is used
    as a template for the appended processing steps description in which "<coords>" will be replaced
    by the coordinates of the affected timeseries and "<var>" will be replaced by the name of the
    data variable it belongs to. The times will be set automatically to a list of the changed
    timepoints.

    Variables without processing information and timeseries whose processing information
    is missing are left untouched, so a dataset without processing information stays a
    dataset without processing information.

    Parameters
    ----------
    old_ds
        Pre-modification dataset including processing information.
    new_ds
        Post-modification dataset. Has to contain the same data variables with the same
        dimensions and coordinates as ``old_ds``, only the values may differ. Processing
        information is taken from here if it is contained, and from ``old_ds`` otherwise.
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
        A copy of ``new_ds`` in which the new processing step is appended to the
        processing information of each affected timeseries.

    Raises
    ------
    ValueError
        If the data variables, dimensions or coordinates of ``old_ds`` and ``new_ds``
        don't match, or if the processing information doesn't match the data it
        describes.
    """
    old_vars = {var for var in old_ds.data_vars if not is_processing_variable(var)}
    new_vars = {var for var in new_ds.data_vars if not is_processing_variable(var)}
    if old_vars != new_vars:
        raise ValueError(
            f"Data variables of old_ds {sorted(str(var) for var in old_vars)!r} and new_ds "
            f"{sorted(str(var) for var in new_vars)!r} don't match."
        )

    result = new_ds.copy()
    for var in sorted(new_vars, key=str):
        name = processing_variable_name(var)
        if name in new_ds:
            processing_infos = new_ds[name]
        elif name in old_ds:
            processing_infos = old_ds[name]
        else:
            continue

        result[name] = add_processing_step_on_change(
            old_da=old_ds[var],
            new_da=new_ds[var],
            processing_infos=processing_infos,
            function=function,
            description_template=description_template.replace("<var>", str(var)),
            source=source,
        )

    return result
