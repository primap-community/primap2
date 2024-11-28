import numpy as np
import pytest
import xarray as xr
import xarray.testing

import primap2.csg
from primap2.csg import FitParameters, StrategyUnableToProcess
from primap2.tests.csg.utils import get_single_ts


@pytest.mark.parametrize(
    "strategy",
    [
        primap2.csg.SubstitutionStrategy(),
        primap2.csg.GlobalLSStrategy(),
        primap2.csg.LocalTrendsStrategy(),
    ],
    ids=lambda x: x.type,
)
def test_strategies_conform(strategy):
    """Test if strategies conform to the protocol for strategies."""
    ts = get_single_ts(data=1.0)
    ts[0] = np.nan
    fill_ts = get_single_ts(data=2.0)
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
    assert "source" not in result_ts.coords.keys()


def test_globalLS_strategy():
    ts = get_single_ts(data=1.0)
    fill_ts = get_single_ts(data=2.0)

    # nothing to fill
    result_ts, result_descriptions = primap2.csg.GlobalLSStrategy(allow_shift=False).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    xr.testing.assert_allclose(ts, result_ts)
    assert len(result_descriptions) == 1
    assert len(result_descriptions[0].time) == 0  # comparison of results fails
    assert result_descriptions[0].description == "no additional data in B"

    # allow_shift = False
    ts[0] = np.nan
    result_ts, result_descriptions = primap2.csg.GlobalLSStrategy(allow_shift=False).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    xr.testing.assert_allclose(get_single_ts(data=1.0), result_ts)
    assert len(result_descriptions) == 1
    assert result_descriptions[0].time == np.array(["1850"], dtype=np.datetime64)
    assert (
        result_descriptions[0].description
        == "filled with least squares matched data from B. Factor=0.500"
    )

    # allow_shift = True
    ts[1:5] = 0.5
    fill_ts[0:5] = 1.5
    result_ts, result_descriptions = primap2.csg.GlobalLSStrategy(allow_shift=True).fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )

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
            allow_shift=True, allow_negative=False
        ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")

    # error for no overlap
    ts[1:5] = np.nan
    fill_ts[5:] = np.nan
    with pytest.raises(StrategyUnableToProcess):
        result_ts, result_descriptions = primap2.csg.GlobalLSStrategy(
            allow_shift=True, allow_negative=False
        ).fill(ts=ts, fill_ts=fill_ts, fill_ts_repr="B")

    # general
    assert "source" not in result_ts.coords.keys()


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
        "times ['1850-01-01T00'] using factor 0.5;"
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
        "times ['1850-01-01T00'] using factor 0.5; "
        "gap 1870-01-01T00 - 1871-01-01T00: filled for "
        "times ['1870-01-01T00' '1871-01-01T00'] using factor 0.5;"
    )

    fill_ts[23:] = fill_ts[23:] * -1
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
        "times ['1850-01-01T00'] using factor 0.5; "
        "gap 1870-01-01T00 - 1871-01-01T00: negative scaling factor - "
        "use fallback degree 0 negative scaling after fallback - "
        "failed to fill gap;"
    )

    ts[1:5] = np.nan
    fill_ts[5:] = np.nan
    with pytest.raises(StrategyUnableToProcess):
        primap2.csg.LocalTrendsStrategy(fit_params=fit_params).fill(
            ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
        )

    # TODO: to test
    #  all fails and fallbacks
    #  fallback if negative

    # general
    assert "source" not in result_ts.coords.keys()
