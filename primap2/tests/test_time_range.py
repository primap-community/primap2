"""Tests for _time_range.py"""

import numpy as np
import pandas as pd
import pytest

from primap2._time_range import TimeRange, time_ranges


def test_time_range_default_step():
    assert TimeRange("1970", "2015").step == np.timedelta64(1, "Y")
    assert TimeRange("2000-01", "2000-12").step == np.timedelta64(1, "M")


def test_time_range_time_points():
    np.testing.assert_array_equal(
        TimeRange("1990", "2000", np.timedelta64(5, "Y")).time_points(),
        np.array(["1990", "1995", "2000"], dtype="datetime64[Y]"),
    )


@pytest.mark.parametrize(
    ("time_range", "expected"),
    [
        (TimeRange("1970", "1970"), "1970"),
        (TimeRange("1970", "2015"), "1970 to 2015"),
        (TimeRange("1990", "2020", np.timedelta64(5, "Y")), "1990 to 2020 every 5 years"),
    ],
)
def test_time_range_str(time_range, expected):
    assert str(time_range) == expected


@pytest.mark.parametrize(
    ("start", "stop", "step"),
    [
        ("2015", "1970", np.timedelta64(1, "Y")),
        ("1970", "2015", np.timedelta64(0, "Y")),
        ("1970", "2015", np.timedelta64(2, "Y")),
    ],
)
def test_time_range_invalid(start, stop, step):
    with pytest.raises(ValueError):
        TimeRange(start, stop, step)


@pytest.mark.parametrize(
    ("time_points", "expected"),
    [
        ([], ()),
        (pd.date_range("1970", "2015", freq="YS"), (TimeRange("1970", "2015"),)),
        (
            pd.date_range("1990", "2020", freq="5YS"),
            (TimeRange("1990", "2020", np.timedelta64(5, "Y")),),
        ),
        (pd.date_range("2000-01", "2000-12", freq="MS"), (TimeRange("2000-01", "2000-12"),)),
        (
            np.array(["2010", "2000", "2001", "2002", "2015"], dtype="datetime64"),
            (
                TimeRange("2000", "2002"),
                TimeRange("2010", "2010"),
                TimeRange("2015", "2015"),
            ),
        ),
    ],
)
def test_time_ranges(time_points, expected):
    result = time_ranges(time_points)
    assert result == expected
    if len(time_points):
        np.testing.assert_array_equal(
            np.concatenate([r.time_points() for r in result]),
            np.sort(np.asarray(time_points, dtype="datetime64")),
        )
