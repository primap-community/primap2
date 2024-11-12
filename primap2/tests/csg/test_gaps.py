import numpy as np
import pandas as pd
import pytest
import xarray as xr

from primap2.csg._strategies.gaps import (
    calculate_boundary_trend,
    calculate_left_boundary_trend,
    calculate_right_boundary_trend,
    get_gaps,
    get_shifted_time_value,
    timeseries_coord_repr,
)
from primap2.tests.utils import assert_aligned_equal


@pytest.fixture
def test_ts() -> xr.DataArray:
    ts = xr.DataArray(
        np.concatenate(
            [
                np.array([np.nan] * 4),
                np.linspace(6, 12, 11),
                np.array([np.nan] * 6),
                np.linspace(15, 4, 4),
                np.array([np.nan] * 14),
                np.linspace(5, 3, 10),
                np.linspace(3, 4, 5),
                np.array([np.nan] * 20),
            ]
        ),
        coords={"time": pd.date_range("1952-01-01", "2025-01-01", freq="YS"), "category": "test"},
        dims="time",
        name="test_ts",
    )
    return ts


def test_get_gaps_left_boundary(test_ts):
    gaps = get_gaps(test_ts.pr.loc[{"time": slice("1952", "1960")}])
    assert len(gaps) == 1
    assert gaps[0].left == np.datetime64("1952-01-01")
    assert gaps[0].right == np.datetime64("1955-01-01")
    assert gaps[0].type == "start"


def test_get_gaps_right_boundary(test_ts):
    gaps = get_gaps(test_ts.pr.loc[{"time": slice("2000", "2025")}])
    assert len(gaps) == 1
    assert gaps[0].left == np.datetime64("2006-01-01")
    assert gaps[0].right == np.datetime64("2025-01-01")
    assert gaps[0].type == "end"


def test_get_gaps_gaps(test_ts):
    gaps = get_gaps(test_ts.pr.loc[{"time": slice("1960", "2002")}])
    assert len(gaps) == 2
    assert gaps[0].left == np.datetime64("1967-01-01")
    assert gaps[0].right == np.datetime64("1972-01-01")
    assert gaps[0].type == "gap"
    assert gaps[1].left == np.datetime64("1977-01-01")
    assert gaps[1].right == np.datetime64("1990-01-01")
    assert gaps[1].type == "gap"


def test_get_gaps_full(test_ts):
    gaps = get_gaps(test_ts)
    assert len(gaps) == 4
    assert gaps[0].left == np.datetime64("1952-01-01")
    assert gaps[0].right == np.datetime64("1955-01-01")
    assert gaps[0].type == "start"
    assert gaps[1].left == np.datetime64("2006-01-01")
    assert gaps[1].right == np.datetime64("2025-01-01")
    assert gaps[1].type == "end"
    assert gaps[2].left == np.datetime64("1967-01-01")
    assert gaps[2].right == np.datetime64("1972-01-01")
    assert gaps[2].type == "gap"
    assert gaps[3].left == np.datetime64("1977-01-01")
    assert gaps[3].right == np.datetime64("1990-01-01")
    assert gaps[3].type == "gap"


# TODO: could use parametrize here
def test_get_shifted_time_value(test_ts):
    original_value = np.datetime64("1955-01-01")
    shifted_value = get_shifted_time_value(test_ts, original_value=original_value, shift=1)
    assert shifted_value == np.datetime64("1956-01-01")
    shifted_value = get_shifted_time_value(test_ts, original_value=original_value, shift=-3)
    assert shifted_value == np.datetime64("1952-01-01")


def test_calculate_left_boundary_trend(test_ts, caplog):
    gaps = get_gaps(test_ts)

    # linear trend for a left boundary
    fit_degree = 1
    # expected result
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1996", "2005")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_value = np.polyval(coeff, 9)
    expected_ts = test_ts.copy()
    expected_ts.pr.loc[{"time": "2005-01-01"}] = expected_value

    trend_ts = calculate_left_boundary_trend(
        test_ts,
        boundary=gaps[1].left,
        fit_degree=fit_degree,
        trend_length=10,
        trend_length_unit="YS",
        min_trend_points=5,
    )

    assert_aligned_equal(trend_ts, expected_ts, rtol=1e-04, equal_nan=True)

    # constant trend for a right boundary
    fit_degree = 0
    # expected result
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_value = np.polyval(coeff, 9)
    expected_ts = test_ts.copy()
    expected_ts.pr.loc[{"time": "2005-01-01"}] = expected_value

    trend_ts = calculate_left_boundary_trend(
        test_ts,
        boundary=gaps[1].left,
        fit_degree=fit_degree,
        trend_length=10,
        trend_length_unit="YS",
        min_trend_points=5,
    )

    assert_aligned_equal(trend_ts, expected_ts, rtol=1e-04, equal_nan=True)

    # test logging if not enough data points
    test_ts.loc[{"time": slice("1997", "2002")}] = (
        test_ts.loc[{"time": slice("1997", "2002")}] * np.nan
    )
    trend_ts = calculate_left_boundary_trend(
        test_ts,
        boundary=gaps[1].left,
        fit_degree=fit_degree,
        trend_length=10,
        trend_length_unit="YS",
        min_trend_points=5,
    )
    assert trend_ts is None

    log_str = (
        "Not enough values to calculate fit for left boundary at "
        "2005-01-01T00:00:00.000000000. fit_degree: 0, trend_length: 10, "
        "trend_length_unit: YS, min_trend_points: 5. Timeseries info: "
        "{'category': 'test'}"
    )

    assert log_str in caplog.text


def test_calculate_right_boundary_trend(test_ts, caplog):
    gaps = get_gaps(test_ts)

    # linear trend for a right boundary
    fit_degree = 1
    # expected result
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1956", "1965")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_value = np.polyval(coeff, 0)
    expected_ts = test_ts.copy()
    expected_ts.pr.loc[{"time": "1956-01-01"}] = expected_value

    trend_ts = calculate_right_boundary_trend(
        test_ts,
        boundary=gaps[0].right,
        fit_degree=fit_degree,
        trend_length=10,
        trend_length_unit="YS",
        min_trend_points=5,
    )

    assert_aligned_equal(trend_ts, expected_ts, rtol=1e-04, equal_nan=True)

    # constant trend for a right boundary
    fit_degree = 0
    # expected result
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_value = np.polyval(coeff, 0)
    expected_ts = test_ts.copy()
    expected_ts.pr.loc[{"time": "1956-01-01"}] = expected_value

    trend_ts = calculate_right_boundary_trend(
        test_ts,
        boundary=gaps[0].right,
        fit_degree=fit_degree,
        trend_length=10,
        trend_length_unit="YS",
        min_trend_points=5,
    )

    assert_aligned_equal(trend_ts, expected_ts, rtol=1e-04, equal_nan=True)

    # test logging if not enough data points
    test_ts.loc[{"time": slice("1958", "1964")}] = (test_ts.loc)[
        {"time": slice("1958", "1964")}
    ] * np.nan
    trend_ts = calculate_right_boundary_trend(
        test_ts,
        boundary=gaps[0].right,
        fit_degree=fit_degree,
        trend_length=10,
        trend_length_unit="YS",
        min_trend_points=5,
    )
    assert trend_ts is None

    log_str = (
        "Not enough values to calculate fit for right boundary at "
        "1956-01-01T00:00:00.000000000. fit_degree: 0, trend_length: 10, "
        "trend_length_unit: YS, min_trend_points: 5. Timeseries info: "
        "{'category': 'test'}"
    )

    assert log_str in caplog.text


def test_calculate_boundary_trend(test_ts):
    # as we test all the individual functions we just construct one big example
    # using a beginning, an end and a gap
    gaps = get_gaps(test_ts)
    fit_degree = 1
    test_ts.pr.loc[{"time": pd.date_range(gaps[2].right, gaps[3].left, freq="YS")}] = (
        test_ts.pr.loc
    )[{"time": pd.date_range(gaps[2].right, gaps[3].left, freq="YS")}] * np.nan
    gaps = get_gaps(test_ts)

    # expected result
    expected_ts = test_ts.copy()
    # beginning
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1956", "1965")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_ts.pr.loc[{"time": "1956-01-01"}] = np.polyval(coeff, 0)
    # end
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1996", "2005")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_ts.pr.loc[{"time": "2005-01-01"}] = np.polyval(coeff, 9)
    # gap left
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1957", "1966")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_ts.pr.loc[{"time": "1966-01-01"}] = np.polyval(coeff, 9)
    # gap right
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1991", "2000")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_ts.pr.loc[{"time": "1991-01-01"}] = np.polyval(coeff, 0)

    # calculate trends (we can do this all on one dataset as the trend intervals don't
    # overlap with calculated trend values)
    trend_ts = calculate_boundary_trend(
        test_ts,
        gap=gaps[0],
        fit_degree=fit_degree,
        trend_length=10,
        trend_length_unit="YS",
        min_trend_points=5,
    )
    trend_ts = calculate_boundary_trend(
        trend_ts,
        gap=gaps[1],
        fit_degree=fit_degree,
        trend_length=10,
        trend_length_unit="YS",
        min_trend_points=5,
    )
    trend_ts = calculate_boundary_trend(
        trend_ts,
        gap=gaps[2],
        fit_degree=fit_degree,
        trend_length=4,
        trend_length_unit="YS",
        min_trend_points=4,
    )

    assert_aligned_equal(trend_ts, expected_ts, rtol=1e-04, equal_nan=True)


def test_timeseries_coords_repr(test_ts):
    test_repr = timeseries_coord_repr(ts=test_ts)
    assert test_repr == "{'category': 'test'}"

    test_ts = test_ts.assign_coords({"area (ISO3)": "TUV", "scenario": "test"})
    test_repr = timeseries_coord_repr(ts=test_ts)
    expected_repr = "{'area (ISO3)': 'TUV', 'category': 'test', 'scenario': 'test'}"
    assert test_repr == expected_repr


def test_temp_polyval():
    test_ts = xr.DataArray(
        np.linspace(6, 12, 11),
        coords={"time": pd.date_range("1956-01-01", "1966-01-01", freq="YS")},
        dims="time",
        name="test_ts",
    )

    time_to_eval = np.datetime64("1957-01-01")

    fit = test_ts.polyfit(dim="time", deg=1, skipna=True)
    value = xr.polyval(
        test_ts.coords["time"].loc[{"time": [time_to_eval]}], fit.polyfit_coefficients
    )

    assert np.allclose(
        test_ts.loc[{"time": [time_to_eval]}].data,
        value.data,
        rtol=1e-03,
    )
