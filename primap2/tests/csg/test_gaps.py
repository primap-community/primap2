import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from primap2.csg._strategies.gaps import (
    FitParameters,
    Gap,
    calculate_boundary_trend,
    calculate_boundary_trend_inner,
    calculate_boundary_trend_with_fallback,
    calculate_scaling_factor,
    fill_gap,
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


@pytest.fixture
def fill_ts(test_ts: xr.DataArray) -> xr.DataArray:
    fill_ts = test_ts.copy()
    fill_ts.pr.loc[{"time": pd.date_range("1956-01-01", "1966-01-01", freq="YS")}] = (
        fill_ts.pr.loc[{"time": pd.date_range("1956-01-01", "1966-01-01", freq="YS")}] * 3
    )
    fill_ts.pr.loc[{"time": pd.date_range("1952-01-01", "1955-01-01", freq="YS")}] = np.linspace(
        7, 6, 4
    )
    fill_ts.pr.loc[{"time": pd.date_range("1967-01-01", "1972-01-01", freq="YS")}] = np.linspace(
        35, 15, 6
    )
    return fill_ts


@pytest.fixture
def expected_ts(test_ts: xr.DataArray) -> xr.DataArray:
    expected_ts = test_ts.copy()
    expected_ts.pr.loc[{"time": pd.date_range("1952-01-01", "1955-01-01", freq="YS")}] = (
        np.linspace(7 / 3, 2, 4)
    )
    factor_ts = np.linspace(1 / 3, 1, 6)
    fill_ts_gap = np.linspace(35, 15, 6)
    expected_ts.pr.loc[{"time": pd.date_range("1967-01-01", "1972-01-01", freq="YS")}] = (
        factor_ts * fill_ts_gap
    )
    return expected_ts


@pytest.fixture
def fit_params_linear() -> FitParameters:
    return FitParameters(
        fit_degree=1,
        trend_length=10,
        trend_length_unit="YS",
        min_trend_points=5,
        fallback_degree=0,
    )


@pytest.fixture
def fit_params_constant() -> FitParameters:
    return FitParameters(
        fit_degree=0,
        trend_length=10,
        trend_length_unit="YS",
        min_trend_points=5,
    )


def test_gap():
    gap = Gap(type="gap", left=np.datetime64("1952-01-01"), right=np.datetime64("2025-01-01"))
    assert gap.get_date_slice() == {
        "time": slice(np.datetime64("1952-01-01"), np.datetime64("2025-01-01"))
    }
    assert gap.type == "gap"
    assert gap.left == np.datetime64("1952-01-01")
    assert gap.right == np.datetime64("2025-01-01")


def test_fit_parameters():
    # check default initialization
    fit_params = FitParameters()
    assert fit_params.fit_degree == 1
    assert fit_params.min_trend_points == 5
    assert fit_params.trend_length == 10
    assert fit_params.fallback_degree == 0
    assert fit_params.trend_length_unit == "YS"

    # check log output
    fit_params = FitParameters(
        fit_degree=2,
        min_trend_points=4,
        trend_length=5,
        fallback_degree=1,
        trend_length_unit="YE",
    )
    assert (
        fit_params.log_string(fallback=True)
        == "fit_degree: 2, trend_length: 5, trend_length_unit: YE, "
        "min_trend_points: 4, fallback_degree: 1.\n"
    )

    assert (
        fit_params.log_string(fallback=False)
        == "fit_degree: 2, trend_length: 5, trend_length_unit: YE, "
        "min_trend_points: 4.\n"
    )

    # check fallback object
    fallback_params = fit_params.get_fallback()
    assert fallback_params.fit_degree == 1
    assert fallback_params.min_trend_points == 1

    # check error message
    with pytest.raises(
        ValueError,
        match=re.escape("min_trend_points (4) must not be smaller than fit_degree (5)."),
    ):
        FitParameters(
            fit_degree=5,
            min_trend_points=4,
            trend_length=5,
            trend_length_unit="YS",
        )


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


def test_get_shifted_time_value(test_ts):
    original_value = np.datetime64("1955-01-01")
    shifted_value = get_shifted_time_value(test_ts, original_value=original_value, shift=1)
    assert shifted_value == np.datetime64("1956-01-01")
    shifted_value = get_shifted_time_value(test_ts, original_value=original_value, shift=-3)
    assert shifted_value == np.datetime64("1952-01-01")


def test_calculate_boundary_trend_inner_left(
    test_ts, fit_params_linear, fit_params_constant, caplog
):
    gaps = get_gaps(test_ts)

    # linear trend for a left boundary
    fit_degree = 1
    # expected result
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1996", "2005")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_value = np.polyval(coeff, 9)
    trend_value = calculate_boundary_trend_inner(
        test_ts,
        side="left",
        boundary=gaps[1].left,
        fit_params=fit_params_linear,
    )
    assert np.allclose(expected_value, trend_value)

    # constant trend for a right boundary
    fit_degree = 0
    # expected result
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_value = np.polyval(coeff, 9)
    trend_value = calculate_boundary_trend_inner(
        test_ts,
        side="left",
        boundary=gaps[1].left,
        fit_params=fit_params_constant,
    )
    assert np.allclose(expected_value, trend_value)

    # test logging if not enough data points
    test_ts.loc[{"time": slice("1997", "2002")}] = (
        test_ts.loc[{"time": slice("1997", "2002")}] * np.nan
    )
    trend_value = calculate_boundary_trend_inner(
        test_ts, side="left", boundary=gaps[1].left, fit_params=fit_params_linear
    )
    assert np.isnan(trend_value)

    log_str = (
        "Not enough values to calculate fit for left boundary at "
        "2005-01-01T00:00:00.000000000.\nfit_degree: 1, trend_length: 10, "
        "trend_length_unit: YS, min_trend_points: 5.\nTimeseries info: "
        "{'category': 'test'}"
    )

    assert log_str in caplog.text


def test_calculate_boundary_trend_inner_right(
    test_ts, fit_params_linear, fit_params_constant, caplog
):
    gaps = get_gaps(test_ts)

    # linear trend for a right boundary
    fit_degree = 1
    # expected result
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1956", "1965")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_value = np.polyval(coeff, 0)

    trend_value = calculate_boundary_trend_inner(
        test_ts,
        side="right",
        boundary=gaps[0].right,
        fit_params=fit_params_linear,
    )

    assert np.allclose(expected_value, trend_value, rtol=1e-04)

    # constant trend for a right boundary
    fit_degree = 0
    # expected result
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_value = np.polyval(coeff, 0)

    trend_value = calculate_boundary_trend_inner(
        test_ts,
        side="right",
        boundary=gaps[0].right,
        fit_params=fit_params_constant,
    )

    assert np.allclose(expected_value, trend_value, rtol=1e-04)

    # test logging if not enough data points
    test_ts.loc[{"time": slice("1958", "1964")}] = (test_ts.loc)[
        {"time": slice("1958", "1964")}
    ] * np.nan
    trend_value = calculate_boundary_trend_inner(
        test_ts,
        side="right",
        boundary=gaps[0].right,
        fit_params=fit_params_linear,
    )
    assert np.isnan(trend_value)

    log_str = (
        "Not enough values to calculate fit for right boundary at "
        "1956-01-01T00:00:00.000000000.\nfit_degree: 1, trend_length: 10, "
        "trend_length_unit: YS, min_trend_points: 5.\nTimeseries info: "
        "{'category': 'test'}"
    )

    assert log_str in caplog.text


def test_calculate_boundary_trend(test_ts, fit_params_linear):
    # as we test all the individual functions we just construct one big example
    # using a beginning, an end and a gap
    gaps = get_gaps(test_ts)
    fit_degree = 1
    test_ts.pr.loc[{"time": pd.date_range(gaps[2].right, gaps[3].left, freq="YS")}] = (
        test_ts.pr.loc
    )[{"time": pd.date_range(gaps[2].right, gaps[3].left, freq="YS")}] * np.nan
    gaps = get_gaps(test_ts)

    # expected results
    # beginning
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1956", "1965")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_start = np.polyval(coeff, 0)
    # end
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1996", "2005")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_end = np.polyval(coeff, 9)
    # gap left
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1957", "1966")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_gap_left = np.polyval(coeff, 9)
    # gap right
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1991", "2000")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=fit_degree)
    expected_gap_right = np.polyval(coeff, 0)

    # calculate trend values
    trend_start = calculate_boundary_trend(
        test_ts,
        gap=gaps[0],
        fit_params=fit_params_linear,
    )
    assert np.allclose(trend_start[1], expected_start, rtol=1e-04)

    trend_end = calculate_boundary_trend(
        test_ts,
        gap=gaps[1],
        fit_params=fit_params_linear,
    )
    assert np.allclose(trend_end[0], expected_end, rtol=1e-04)

    fit_params_short = FitParameters(
        fit_degree=fit_degree,
        trend_length=4,
        trend_length_unit="YS",
        min_trend_points=4,
    )

    trend_gap = calculate_boundary_trend(
        test_ts,
        gap=gaps[2],
        fit_params=fit_params_short,
    )

    assert np.allclose(trend_gap, [expected_gap_left, expected_gap_right], rtol=1e-04)


def test_calculate_boundary_trend_with_fallback(test_ts, fit_params_linear):
    gaps = get_gaps(test_ts)

    # expected result linear
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1956", "1965")}].data
    coeff = np.polyfit(range(0, 10), data_to_interpolate, deg=1)
    value = np.polyval(coeff, 0)
    expected_linear = [value, value]

    trend_values = calculate_boundary_trend_with_fallback(
        test_ts,
        gap=gaps[0],
        fit_params=fit_params_linear,
    )
    assert np.allclose(trend_values, expected_linear, rtol=1e-04)

    # remove some data points for fallback
    test_ts.loc[{"time": slice("1958", "1964")}] = (
        test_ts.loc[{"time": slice("1958", "1964")}] * np.nan
    )
    # expected result constant
    data_to_interpolate = test_ts.pr.loc[{"time": slice("1956", "1965")}].data
    idx_notna = np.isfinite(data_to_interpolate)
    x_vals = np.array(range(0, 10))
    coeff = np.polyfit(x_vals[idx_notna], data_to_interpolate[idx_notna], deg=0)
    value = np.polyval(coeff, 0)
    expected_constant = [value, value]

    trend_values = calculate_boundary_trend_with_fallback(
        test_ts,
        gap=gaps[0],
        fit_params=fit_params_linear,
    )
    assert np.allclose(trend_values, expected_constant, rtol=1e-04)

    # test filling of missing boundary trends for a gap
    # this can only occur when we use gap information for one time-series on another
    # time-series, so we fake a gap here
    fake_gap = Gap(left=np.datetime64("1953-01-01"), right=np.datetime64("1955-01-01"), type="gap")
    trend_values = calculate_boundary_trend_with_fallback(
        test_ts,
        gap=fake_gap,
        fit_params=fit_params_linear,
    )
    assert np.allclose(trend_values, expected_constant, rtol=1e-04)


def test_calculate_scaling_factor(test_ts, fill_ts, fit_params_linear, caplog):
    gaps = get_gaps(test_ts)

    factor = calculate_scaling_factor(
        ts=test_ts, fill_ts=fill_ts, fit_params=fit_params_linear, gap=gaps[2]
    )
    assert np.allclose([0.33333, 1], factor)
    # as fallback will be used we also check for the log message (not strictly necessary
    # to test his as it's raise by a different function)
    assert (
        "Not enough values to calculate fit for right boundary "
        "at 1973-01-01T00:00:00.000000000.\nfit_degree: 1, trend_length: 10, "
        "trend_length_unit: YS, min_trend_points: 5.\n"
        "Timeseries info: {'category': 'test'}" in caplog.text
    )

    # zero for fill_ts only result should be nan
    fill_ts.pr.loc[{"time": pd.date_range("1956-01-01", "1966-01-01", freq="YS")}] = (
        fill_ts.pr.loc[{"time": pd.date_range("1956-01-01", "1966-01-01", freq="YS")}] * 0
    )
    factor = calculate_scaling_factor(
        ts=test_ts, fill_ts=fill_ts, fit_params=fit_params_linear, gap=gaps[0]
    )
    assert all(np.isinf(factor))

    # zero for both ts: result should be 0
    test_ts.pr.loc[{"time": pd.date_range("1956-01-01", "1966-01-01", freq="YS")}] = (
        test_ts.pr.loc[{"time": pd.date_range("1956-01-01", "1966-01-01", freq="YS")}] * 0
    )
    factor = calculate_scaling_factor(
        ts=test_ts, fill_ts=fill_ts, fit_params=fit_params_linear, gap=gaps[0]
    )
    assert np.allclose(factor, [0, 0])


def test_fill_gap(test_ts, fill_ts, expected_ts, fit_params_linear):
    gaps = get_gaps(test_ts)

    # to test
    # fill gap at end or beginning
    factor = calculate_scaling_factor(
        ts=test_ts, fill_ts=fill_ts, fit_params=fit_params_linear, gap=gaps[0]
    )
    filled_ts = fill_gap(ts=test_ts, fill_ts=fill_ts, gap=gaps[0], factor=factor)

    # fill gap in the middle
    factor = calculate_scaling_factor(
        ts=test_ts, fill_ts=fill_ts, fit_params=fit_params_linear, gap=gaps[2]
    )
    filled_ts = fill_gap(ts=filled_ts, fill_ts=fill_ts, gap=gaps[2], factor=factor)

    assert_aligned_equal(expected_ts, filled_ts, rtol=1e-03, equal_nan=True)


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
