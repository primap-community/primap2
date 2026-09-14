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


def add_processing_step(da: xr.DataArray, step: ProcessingStepDescription) -> xr.DataArray:
    """Append a processing step to every timeseries of the input.

    Timeseries whose processing information is missing are left untouched, because
    appending a step to them would claim that this step is the only processing which was
    done to them, while in fact nothing is known about their processing.

    Parameters
    ----------
    da
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

    result = da.copy()
    result.data = np.vectorize(append, otypes=[object])(da.data)
    return result
