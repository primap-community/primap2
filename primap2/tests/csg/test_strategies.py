import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2.csg
from primap2.csg import FitParameters, MatchParameters, StrategyUnableToProcess
from primap2.csg._strategies.gaps import get_matching_index
from primap2.tests.csg.utils import get_single_ts


@pytest.mark.parametrize(
    "strategy",
    [
        primap2.csg.SubstitutionStrategy(),
        primap2.csg.GlobalLSStrategy(),
        primap2.csg.LocalLSStrategy(),
        primap2.csg.LocalTrendsStrategy(),
    ],
    ids=lambda x: x.type,
)
def test_strategies_conform(strategy):
    """Test if strategies conform to the protocol for strategies."""
    ts = get_single_ts(data=1.0)
    ts[0] = np.nan
    fill_ts = get_single_ts(data=2.0)
    # change one data point so the solution for least squares matching is unique
    ts[10] = 1.1
    fill_ts[10] = 2.2
    initial_ts = ts.copy(deep=True)
    initial_fill_ts = fill_ts.copy(deep=True)

    res, res_desc = strategy.fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")

    # ts and fill_ts must not be modified
    xr.testing.assert_identical(ts, initial_ts)
    xr.testing.assert_identical(fill_ts, initial_fill_ts)

    assert isinstance(res, xr.DataArray)
    assert isinstance(res_desc, list)
    assert all(isinstance(step, primap2.ProcessingStepDescription) for step in res_desc)


def test_substitution_strategy():
    ts = get_single_ts(data=1.0)
    ts[0] = np.nan
    fill_ts = get_single_ts(data=2.0)

    result_ts, result_descriptions = primap2.csg.SubstitutionStrategy().fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    assert result_ts[0] == 2.0
    assert (result_ts[1:] == 1.0).all()
    assert len(result_descriptions) == 1
    assert result_descriptions[0].time == np.array(["1850"], dtype=np.datetime64)
    assert result_descriptions[0].description == "substituted with corresponding values from B"
    assert "source" not in result_ts.coords


def test_globalLS_strategy():
    ts = get_single_ts(data=1.0)
    fill_ts = get_single_ts(data=2.0)

    match_params = MatchParameters(
        fit_length=20,
        fit_length_unit="YS",
        allowed_factor=2.01,  # 0.1 to avoid fails due to rounding
    )

    # nothing to fill
    result_ts, result_descriptions = primap2.csg.GlobalLSStrategy(
        match_params=match_params, allow_shift=False
    ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")
    xr.testing.assert_allclose(ts, result_ts)
    assert len(result_descriptions) == 1
    assert len(result_descriptions[0].time) == 0  # comparison of results fails
    assert result_descriptions[0].description == "no additional data in B"

    # allow_shift = False
    ts[0] = np.nan
    result_ts, result_descriptions = primap2.csg.GlobalLSStrategy(
        match_params=match_params, allow_shift=False
    ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")
    xr.testing.assert_allclose(get_single_ts(data=1.0), result_ts)
    assert len(result_descriptions) == 1
    assert result_descriptions[0].time == np.array(["1850"], dtype=np.datetime64)
    assert (
        result_descriptions[0].description
        == "filled with least squares matched data from B. Factor=0.500"
    )

    # allow_shift = True
    match_params = MatchParameters(
        fit_length=20,
        fit_length_unit="YS",
        allowed_factor=2.01,  # 0.1 to avoid fails due to rounding
        allowed_shift=1.001,
    )
    ts[1:5] = 0.5
    fill_ts[0:5] = 1.5
    result_ts, result_descriptions = primap2.csg.GlobalLSStrategy(
        match_params=match_params, allow_shift=True
    ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")

    expected_ts = ts.copy(deep=True)
    expected_ts[0] = 0.5
    xr.testing.assert_allclose(expected_ts, result_ts)
    assert len(result_descriptions) == 1
    assert result_descriptions[0].time == np.array(["1850"], dtype=np.datetime64)
    assert (
        result_descriptions[0].description == "filled with least squares matched data from B. "
        "a*x+b with a=1.000, b=-1.000"
    )

    # error for negative emissions
    ts[1:5] = 0.5
    fill_ts[1:5] = 1.5
    fill_ts[0] = 0.1
    with pytest.raises(StrategyUnableToProcess):
        result_ts, result_descriptions = primap2.csg.GlobalLSStrategy(
            match_params=match_params, allow_shift=True, allow_negative=False
        ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")

    # error for no overlap
    ts[1:5] = np.nan
    fill_ts[5:] = np.nan
    with pytest.raises(StrategyUnableToProcess):
        result_ts, result_descriptions = primap2.csg.GlobalLSStrategy(
            match_params=match_params, allow_shift=True, allow_negative=False
        ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")

    # general
    assert "source" not in result_ts.coords


def test_localTrends_strategy():
    ts = get_single_ts(data=1.0)
    fill_ts = get_single_ts(data=2.0)

    # nothing to fill
    fit_params = FitParameters(
        trend_length=10,
        min_trend_points=5,
        trend_length_unit="YS",
        fit_degree=1,
        fallback_degree=0,
    )
    # allow_negative: bool = False
    result_ts, result_descriptions = primap2.csg.LocalTrendsStrategy(fit_params=fit_params).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    xr.testing.assert_allclose(ts, result_ts)
    assert len(result_descriptions) == 1
    assert len(result_descriptions[0].time) == 0  # comparison of results fails
    assert result_descriptions[0].description == "no additional data in B"

    # fill a gap at the start
    ts[0] = np.nan
    result_ts, result_descriptions = primap2.csg.LocalTrendsStrategy(fit_params=fit_params).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    xr.testing.assert_allclose(get_single_ts(data=1.0), result_ts)
    assert len(result_descriptions) == 1
    assert result_descriptions[0].time == np.array(["1850"], dtype=np.datetime64)
    assert (
        result_descriptions[0].description == "filled with local trend matched data from B. "
        "The following gaps have been filled: "
        "gap 1850-01-01T00 - 1850-01-01T00: filled for "
        "times ['1850-01-01T00'] using factor 0.50;"
    )

    ts[20:22] = np.nan
    result_ts, result_descriptions = primap2.csg.LocalTrendsStrategy(fit_params=fit_params).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    xr.testing.assert_allclose(get_single_ts(data=1.0), result_ts)
    assert len(result_descriptions) == 1
    assert all(
        result_descriptions[0].time == np.array(["1850", "1870", "1871"], dtype=np.datetime64)
    )
    assert (
        result_descriptions[0].description == "filled with local trend matched data from B. "
        "The following gaps have been filled: "
        "gap 1850-01-01T00 - 1850-01-01T00: filled for "
        "times ['1850-01-01T00'] using factor 0.50; "
        "gap 1870-01-01T00 - 1871-01-01T00: filled for "
        "times ['1870-01-01T00' '1871-01-01T00'] using factor 0.50;"
    )

    fill_ts[22:] = fill_ts[22:] * -1
    result_ts, result_descriptions = primap2.csg.LocalTrendsStrategy(fit_params=fit_params).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    expected_ts = ts.copy()
    expected_ts[0] = 1
    xr.testing.assert_allclose(expected_ts, result_ts)
    assert all(result_descriptions[0].time == np.array(["1850"], dtype=np.datetime64))
    assert (
        result_descriptions[0].description == "filled with local trend matched data from B. "
        "The following gaps have been filled: "
        "gap 1850-01-01T00 - 1850-01-01T00: filled for "
        "times ['1850-01-01T00'] using factor 0.50; "
        "gap 1870-01-01T00 - 1871-01-01T00: negative scaling factor - "
        "use fallback degree 0 negative scaling after fallback - "
        "failed to fill gap;"
    )

    #  gap description for differing scaling factors
    fill_ts[22:] = fill_ts[22:] * -1
    ts[0] = 1
    ts[22:] = ts[22:] * 3
    result_ts, result_descriptions = primap2.csg.LocalTrendsStrategy(fit_params=fit_params).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    expected_ts = ts.copy()
    expected_ts[20] = 1
    expected_ts[21] = 3
    xr.testing.assert_allclose(expected_ts, result_ts)
    assert all(result_descriptions[0].time == np.array(["1870", "1871"], dtype=np.datetime64))
    assert (
        result_descriptions[0].description == "filled with local trend matched data from B. "
        "The following gaps have been filled: "
        "gap 1870-01-01T00 - 1871-01-01T00: filled for "
        "times ['1870-01-01T00' '1871-01-01T00'] using factors 0.50 and 1.50;"
    )

    # remove some data points from fill_ts completely (including index)
    # this is not possible in normal csg usage of the function as all
    # time-series come from the same dataset.
    fill_ts_temp = fill_ts.drop_sel(time=["1871-01-01", "1872-01-01"])
    expected_msg = "Index 1871-01-01 not found in timeseries."

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        result_ts, result_descriptions = primap2.csg.LocalTrendsStrategy(
            fit_params=fit_params
        ).fill(ts=ts, fill_ts=fill_ts_temp, fill_ts_repr="B")

    ts[0:5] = np.nan
    fill_ts[5:] = np.nan
    with pytest.raises(StrategyUnableToProcess):
        primap2.csg.LocalTrendsStrategy(fit_params=fit_params).fill(
            ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # general
    assert "source" not in result_ts.coords


def test_MatchParameters():
    expected_msg = "allowed_factor (0.5) has to be larger than 1.0 or equal to 0.0."
    with pytest.raises(
        ValueError,
        match=re.escape(expected_msg),
    ):
        MatchParameters(allowed_factor=0.5)


def test_localLS_variance():
    time = pd.date_range(
        start=pd.to_datetime("1990"),
        end=pd.to_datetime("2022"),
        freq="YS",
    )

    da = get_single_ts(time=time, data=range(33))
    da.data = da.data.astype("float")
    da[3] = np.nan
    da[-1] = np.nan

    da_var = primap2.csg.GlobalLSStrategy()._variance_detrend(ts=da)
    assert np.allclose(da_var.data, np.array(0.86949974))


def test_get_matching_index():
    ts = get_single_ts(data=1.0)
    fill_ts = get_single_ts(data=2.0)

    ts[0:10] = np.nan
    ts[-10:-1] = np.nan
    ts[13:20] = np.nan
    match_params = MatchParameters(
        fit_length=5,
        fit_length_unit="YS",
    )

    index = get_matching_index(
        ts=ts,
        fill_ts=fill_ts,
        side="right",
        boundary=ts.coords["time"].values[9],
        match_params=match_params,
    )
    expected = pd.DatetimeIndex(
        ["1860-01-01", "1861-01-01", "1862-01-01"], dtype="datetime64[us]", freq="YS-JAN"
    )
    pd.testing.assert_index_equal(index, expected)

    index = get_matching_index(
        ts=ts,
        fill_ts=fill_ts,
        side="left",
        boundary=ts.coords["time"].values[13],
        match_params=match_params,
    )
    expected = pd.DatetimeIndex(
        ["1860-01-01", "1861-01-01", "1862-01-01"], dtype="datetime64[us]", freq="YS-JAN"
    )
    pd.testing.assert_index_equal(index, expected)

    index = get_matching_index(
        ts=ts,
        fill_ts=fill_ts,
        side="right",
        boundary=ts.coords["time"].values[19],
        match_params=match_params,
    )
    expected = pd.DatetimeIndex(
        ["1870-01-01", "1871-01-01", "1872-01-01", "1873-01-01", "1874-01-01"],
        dtype="datetime64[us]",
        freq="YS-JAN",
    )
    pd.testing.assert_index_equal(index, expected)

    # check if empty index is returned when there are no data for matching in fill_ts
    fill_ts_temp = get_single_ts(data=2.0)
    fill_ts_temp[9:] = np.nan
    index = get_matching_index(
        ts=ts,
        fill_ts=fill_ts_temp,
        side="right",
        boundary=ts.coords["time"].values[19],
        match_params=match_params,
    )
    expected = pd.DatetimeIndex([], dtype="datetime64[us]", freq="YS-JAN")
    pd.testing.assert_index_equal(index, expected)

    # remove data points completely (also from the index)
    fill_ts = fill_ts.drop_sel(time=["1871-01-01", "1872-01-01"])
    index = get_matching_index(
        ts=ts,
        fill_ts=fill_ts,
        side="right",
        boundary=ts.coords["time"].values[19],
        match_params=match_params,
    )
    expected = pd.DatetimeIndex(["1870-01-01", "1873-01-01", "1874-01-01"], dtype="datetime64[us]")
    pd.testing.assert_index_equal(index, expected)


def test_localLS_strategy():
    ts = get_single_ts(data=1.0)
    fill_ts = get_single_ts(data=2.0)
    # change one data point for a unique solution of LS fitting
    ts[10] = ts[10] * 0.2
    fill_ts[10] = fill_ts[10] * 0.2

    # nothing to fill
    match_params = MatchParameters(
        fit_length=20,
        fit_length_unit="YS",
        allowed_factor=2.01,  # 0.1 to avoid fails due to rounding
    )

    result_ts, result_descriptions = primap2.csg.LocalLSStrategy(
        match_params=match_params, allow_shift=False
    ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")
    xr.testing.assert_allclose(ts, result_ts)
    assert len(result_descriptions) == 1
    assert len(result_descriptions[0].time) == 0  # comparison of results fails
    assert result_descriptions[0].description == "no additional data in B"

    # fill a gap at the start
    ts[0] = np.nan
    result_ts, result_descriptions = primap2.csg.LocalLSStrategy(
        match_params=match_params, allow_shift=False
    ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")
    expected_ts = get_single_ts(data=1.0)
    expected_ts[10] = expected_ts[10] * 0.2
    xr.testing.assert_allclose(expected_ts, result_ts)
    assert len(result_descriptions) == 1
    assert result_descriptions[0].time == np.array(["1850"], dtype=np.datetime64)
    assert (
        result_descriptions[0].description == "The following gaps have been filled: "
        "gap 1850-01-01T00 - 1850-01-01T00: filled for times ['1850-01-01T00']; "
        "Gaps filled with local least squares matched data from B. "
        "Factor=0.500, "
    )

    ts[20:22] = np.nan
    result_ts, result_descriptions = primap2.csg.LocalLSStrategy(
        match_params=match_params, allow_shift=False
    ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")
    xr.testing.assert_allclose(expected_ts, result_ts)
    assert len(result_descriptions) == 1
    assert all(
        result_descriptions[0].time == np.array(["1850", "1870", "1871"], dtype=np.datetime64)
    )
    assert (
        result_descriptions[0].description == "The following gaps have been filled: "
        "gap 1850-01-01T00 - 1850-01-01T00: filled for times ['1850-01-01T00'];  "
        "gap 1870-01-01T00 - 1871-01-01T00: filled for times "
        "['1870-01-01T00' '1871-01-01T00']; "
        "Gaps filled with local least squares matched data from B. "
        "Factor=0.500, "
    )

    #  test with shift
    ts[22:] = ts[22:] * 3
    fill_ts[22:] = fill_ts[22:] * 3
    fill_ts.data = fill_ts.data + 0.1
    result_ts, result_descriptions = primap2.csg.LocalLSStrategy(
        match_params=match_params, allow_shift=True
    ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")
    expected_ts = ts.copy()
    expected_ts[0] = 1
    expected_ts[20:22] = 1
    xr.testing.assert_allclose(expected_ts, result_ts)
    assert all(
        result_descriptions[0].time == np.array(["1850", "1870", "1871"], dtype=np.datetime64)
    )
    assert (
        result_descriptions[0].description
        == "The following gaps have been filled: gap 1850-01-01T00 - 1850-01-01T00: "
        "filled for times ['1850-01-01T00'];  gap 1870-01-01T00 - 1871-01-01T00: "
        "filled for times ['1870-01-01T00' '1871-01-01T00']; Gaps filled with local "
        "least squares matched data from B. a*x+b with a=0.500, b=-0.050"
    )

    #  test if changes outside the matching period influence the results
    fill_ts[43:] = fill_ts[43:] * 0.5
    result_ts_2, result_descriptions = primap2.csg.LocalLSStrategy(
        match_params=match_params, allow_shift=True
    ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")
    xr.testing.assert_allclose(result_ts, result_ts_2)

    # test if strategy fails when there is no overlap
    ts[0:5] = np.nan
    fill_ts[5:] = np.nan
    with pytest.raises(StrategyUnableToProcess):
        primap2.csg.LocalLSStrategy(match_params=match_params).fill(
            ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # general
    assert "source" not in result_ts.coords


def test_localLS_strategy_allow_negative():
    ts = get_single_ts(data=1.0)
    ts[0] = np.nan
    ts[20:22] = np.nan
    ts_neg = get_single_ts(data=-1.0)
    ts_neg[0] = np.nan
    ts_neg[20:22] = np.nan
    fill_ts = get_single_ts(data=2.0)
    # change one data point for a unique solution of LS fitting
    ts[10] = ts[10] * 0.2
    ts_neg[10] = ts_neg[10] * 0.2
    fill_ts[10] = fill_ts[10] * 0.2

    # nothing to fill
    match_params = MatchParameters(
        fit_length=20,
        fit_length_unit="YS",
        allowed_factor=3.01,  # 0.1 to avoid fails due to rounding
    )

    # test if strategy fails because of negative data
    expected_msg = "Negative data after harmonization excluded by configuration"
    with pytest.raises(
        StrategyUnableToProcess,
        match=re.escape(expected_msg),
    ):
        primap2.csg.LocalLSStrategy(match_params=match_params).fill(
            ts=ts_neg, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # test if strategy works with negative data when `allow_negative=True`
    fill_ts[0] = fill_ts[0] * -1
    result_ts, result_descriptions = primap2.csg.LocalLSStrategy(
        match_params=match_params, allow_shift=False, allow_negative=True
    ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")
    expected_ts = ts.copy()
    expected_ts[0] = -1
    expected_ts[20:22] = 1
    xr.testing.assert_allclose(expected_ts, result_ts)
    assert all(
        result_descriptions[0].time == np.array(["1850", "1870", "1871"], dtype=np.datetime64)
    )
    assert (
        result_descriptions[0].description == "The following gaps have been filled: "
        "gap 1850-01-01T00 - 1850-01-01T00: filled for times ['1850-01-01T00'];  "
        "gap 1870-01-01T00 - 1871-01-01T00: filled for times "
        "['1870-01-01T00' '1871-01-01T00']; "
        "Gaps filled with local least squares matched data from B. "
        "Factor=0.500, "
    )


def test_localLS_strategy_residual():
    ts = get_single_ts(data=1.0)
    fill_ts = get_single_ts(data=2.0)
    ts[0] = np.nan
    ts[20:22] = np.nan
    fill_ts[22:] = fill_ts[22:] * 2
    ts[0:10] = ts[0:10] * 3

    # test if strategy fails when residual is too high
    match_params = MatchParameters(
        fit_length=20,
        fit_length_unit="YS",
        var_average_period=11,
        allowed_res=1,
        allowed_factor=3,
        allowed_shift=3,
    )

    # with shift allowed
    expected_msg = (
        "Residual higher than allowed: "
        f"sqrt(normed residual)/max(sigma(ts), sigma(fill_ts) "
        f"= 2.280 sigma > {match_params.allowed_res} sigma."
    )
    with pytest.raises(
        StrategyUnableToProcess,
        match=re.escape(expected_msg),
    ):
        primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=True).fill(
            ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # with shift not allowed
    expected_msg = (
        "Residual higher than allowed: "
        f"sqrt(normed residual)/max(sigma(ts), sigma(fill_ts) "
        f"= 3.746 sigma > {match_params.allowed_res} sigma."
    )
    with pytest.raises(
        StrategyUnableToProcess,
        match=re.escape(expected_msg),
    ):
        primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=False).fill(
            ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # check if no error is raised when allowed_res=0
    match_params = MatchParameters(
        fit_length=20,
        fit_length_unit="YS",
        var_average_period=11,
        allowed_res=0,
        allowed_factor=3,
        allowed_shift=3,
    )

    # with shift allowed
    primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=True).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    # with shift not allowed
    primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=False).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )


def test_localLS_strategy_factor():
    #  test if strategy fails when factor is too high
    ts = get_single_ts(data=2.0)
    fill_ts = get_single_ts(data=1.0)
    ts[0] = np.nan
    ts[20:22] = np.nan
    # change one data point in both time-series to make the solution unique
    ts[10] = 4
    fill_ts[10] = 2

    match_params = MatchParameters(
        fit_length=20,
        fit_length_unit="YS",
        var_average_period=11,
        allowed_res=1,
        allowed_factor=1.2,
    )

    # with shift allowed
    expected_msg = (
        f"Harmonization factor a is larger than allowed: a=2.000 > {match_params.allowed_factor}"
    )
    with pytest.raises(
        StrategyUnableToProcess,
        match=re.escape(expected_msg),
    ):
        primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=True).fill(
            ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # with shift not allowed
    expected_msg = (
        f"Harmonization factor a is larger than allowed: a=2.000 > {match_params.allowed_factor}"
    )
    with pytest.raises(
        StrategyUnableToProcess,
        match=re.escape(expected_msg),
    ):
        primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=False).fill(
            ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # the same for the inverse message
    ts = get_single_ts(data=1.0)
    fill_ts = get_single_ts(data=2.0)
    ts[0] = np.nan
    ts[20:22] = np.nan
    # change one data point in both time-series to make the solution unique
    ts[10] = 2
    fill_ts[10] = 4

    # with shift allowed
    expected_msg = (
        f"Harmonization factor a is smaller than allowed: "
        f"a=0.500 < {1 / match_params.allowed_factor:.3f}"
    )
    with pytest.raises(
        StrategyUnableToProcess,
        match=re.escape(expected_msg),
    ):
        primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=True).fill(
            ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # with shift not allowed
    expected_msg = (
        f"Harmonization factor a is smaller than allowed: "
        f"a=0.500 < {1 / match_params.allowed_factor:.3f}"
    )
    with pytest.raises(
        StrategyUnableToProcess,
        match=re.escape(expected_msg),
    ):
        primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=False).fill(
            ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # check if no error is raised when allowed_factor=0
    match_params = MatchParameters(
        fit_length=20,
        fit_length_unit="YS",
        var_average_period=11,
        allowed_res=1,
        allowed_factor=0,
    )
    # with shift allowed
    primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=True).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    # with shift not allowed
    primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=False).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )


def test_localLS_strategy_shift():
    #  test if strategy fails when factor is too high
    ts = get_single_ts(data=2.0)
    # change one data point in both time-series to make the solution unique
    ts[10] = 4
    fill_ts = ts.copy()
    fill_ts = fill_ts + 2
    ts[0] = np.nan
    ts[20:22] = np.nan

    match_params = MatchParameters(
        fit_length=20,
        fit_length_unit="YS",
        var_average_period=11,
        allowed_res=1,
        allowed_shift=0.2,
    )

    # with shift allowed
    expected_msg = (
        f"Harmonization shift a is larger than allowed: |b|=2.000 > {match_params.allowed_shift}"
    )
    with pytest.raises(
        StrategyUnableToProcess,
        match=re.escape(expected_msg),
    ):
        primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=True).fill(
            ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # check that no error is raised when allowed_shift=0
    match_params = MatchParameters(
        fit_length=20,
        fit_length_unit="YS",
        var_average_period=11,
        allowed_res=1,
        allowed_shift=0,
    )
    primap2.csg.LocalLSStrategy(match_params=match_params, allow_shift=True).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )


def test_raises_error_substitution_strategy_missing_years():
    ts = get_single_ts(data=1.0)
    # set the year 1850 to NaN
    ts[0] = np.nan
    fill_ts = get_single_ts(data=2.0)
    # remove one year so the shapes are not aligned anymore
    fill_ts = fill_ts.sel(time=slice("1870-01-01", "2020-01-01"))

    expected_msg = (
        "cannot align objects with join='exact' where index/labels/sizes are "
        "not equal along these coordinates (dimensions): 'time' ('time',)"
    )

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        primap2.csg.SubstitutionStrategy().fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")


def test_raises_error_substitution_strategy_wrong_dimension():
    ts = get_single_ts(data=1.0)
    ts[0] = np.nan
    fill_ts = get_single_ts(data=2.0)

    # rename coordinate so the dimensions are not aligned anymore
    fill_ts = fill_ts.rename({"time": "TIME"})

    with pytest.raises(IndexError):
        primap2.csg.SubstitutionStrategy().fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")
