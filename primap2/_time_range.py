"""Compact description of equally spaced time points."""

import typing

import numpy as np
from attr import define, field


@define(frozen=True)
class TimeRange:
    """Equally spaced time points from ``start`` to ``stop``, both included.

    Attributes
    ----------
    start
        The first time point.
    stop
        The last time point.
    step
        The distance between consecutive time points. Defaults to one unit of the
        precision of ``start`` and ``stop``, e.g. one year for
        ``TimeRange("1970", "2015")`` and one month for ``TimeRange("2000-01", "2000-12")``.
    """

    start: np.datetime64 = field(converter=np.datetime64)
    stop: np.datetime64 = field(converter=np.datetime64)
    step: np.timedelta64 = field(converter=np.timedelta64)

    @step.default
    def _step_default(self) -> np.timedelta64:
        unit, _ = np.datetime_data((self.stop - self.start).dtype)
        return np.timedelta64(1, unit)

    def __attrs_post_init__(self):
        if self.step.astype(np.int64) <= 0:
            raise ValueError(f"step has to be positive, not {self.step}.")
        if self.stop < self.start:
            raise ValueError(f"stop {self.stop} is before start {self.start}.")
        if (self.stop - self.start) % self.step:
            raise ValueError(
                f"stop {self.stop} is not reachable from start {self.start} in steps of "
                f"{self.step}."
            )

    def __str__(self) -> str:
        start = np.datetime_as_string(self.start)
        if self.start == self.stop:
            return start
        result = f"{start} to {np.datetime_as_string(self.stop)}"
        unit, _ = np.datetime_data(self.step.dtype)
        if self.step != np.timedelta64(1, unit):
            result += f" every {self.step}"
        return result

    def time_points(self) -> np.ndarray:
        """All time points of the range."""
        return np.arange(self.start, self.stop + self.step, self.step)

    def unstructure(self) -> list[typing.Any]:
        """Convert into basic python types."""
        unit, _ = np.datetime_data(self.step.dtype)
        return [
            np.datetime_as_string(self.start),
            np.datetime_as_string(self.stop),
            int(self.step / np.timedelta64(1, unit)),
            unit,
        ]

    @classmethod
    def structure(cls, u: typing.Sequence[typing.Any]) -> "TimeRange":
        """Initialize from basic python types as created by "unstructure"."""
        start, stop, step, unit = u
        return cls(start=start, stop=stop, step=np.timedelta64(step, unit))


_TIME_UNITS = ("Y", "M", "D", "h", "m", "s", "ms", "us", "ns")


def time_ranges(time_points: typing.Iterable[typing.Any]) -> tuple[TimeRange, ...]:
    """Describe time points compactly as a sorted tuple of TimeRanges.

    The coarsest precision which represents all time points exactly is used, and runs of
    at least three equally spaced time points are combined into one TimeRange.
    """
    points = np.unique(np.asarray(time_points, dtype="datetime64"))
    if np.isnat(points).any():
        raise ValueError("Time points must not contain NaT.")
    if not len(points):
        return ()
    for unit in _TIME_UNITS:
        converted = points.astype(f"datetime64[{unit}]")
        if (converted == points).all():
            break
    values = converted.astype(np.int64)

    result = []
    i = 0
    while i < len(values):
        j = i
        if i + 2 < len(values):
            step = values[i + 1] - values[i]
            while j + 1 < len(values) and values[j + 1] - values[j] == step:
                j += 1
        if j - i < 2:
            # two time points are easier to read as single time points than as a range
            j = i
            step = 1
        result.append(
            TimeRange(start=converted[i], stop=converted[j], step=np.timedelta64(int(step), unit))
        )
        i = j + 1
    return tuple(result)
