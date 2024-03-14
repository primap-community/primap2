"""Tests for csg/test_compose.py"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2.csg._compose


def test_timeseries_selector():
    time = pd.date_range("1850-01-01", "2022-01-01", freq="YS")
    anp = np.linspace(0.0, 1.0, len(time))
    da = xr.DataArray(
        anp, dims=["time"], coords={"time": time, "source": "A", "category": "1.A"}
    )

    assert primap2.csg._compose.TimeseriesSelector({"source": "A"}).match(da)
    assert not primap2.csg._compose.TimeseriesSelector({"source": "B"}).match(da)
    assert primap2.csg._compose.TimeseriesSelector(
        {"source": "A", "category": "1.A"}
    ).match(da)
    assert not primap2.csg._compose.TimeseriesSelector(
        {"source": "A", "category": "1"}
    ).match(da)


def test_strategy_definition():
    time = pd.date_range("1850-01-01", "2022-01-01", freq="YS")
    anp = np.linspace(0.0, 1.0, len(time))
    da = xr.DataArray(
        anp, dims=["time"], coords={"time": time, "source": "A", "category": "1.A"}
    )

    ts = primap2.csg._compose.TimeseriesSelector
    assert (
        primap2.csg._compose.StrategyDefinition(
            [(ts({"source": "A", "category": "1"}), 1), (ts({"source": "A"}), 2)]
        ).find_strategy(da)
        == 2
    )
    assert (
        primap2.csg._compose.StrategyDefinition(
            [
                (ts({"source": "A", "category": "1"}), 1),
                (ts({"source": "A", "category": "1.A"}), 2),
            ]
        ).find_strategy(da)
        == 2
    )
    with pytest.raises(KeyError):
        primap2.csg._compose.StrategyDefinition(
            [
                (ts({"source": "A", "category": "1"}), 1),
                (ts({"source": "A", "category": "1.B"}), 2),
                (ts({"source": "B", "category": "1.B"}), 3),
            ]
        ).find_strategy(da)


def test_compose_timeseries_trivial():
    priority_definition = primap2.csg._compose.PriorityDefinition(
        dimensions=["source"], priorities=[{"source": "A"}, {"source": "B"}]
    )
    strategy_definition = primap2.csg._compose.StrategyDefinition(
        strategies=[
            (
                primap2.csg._compose.TimeseriesSelector({"source": "B"}),
                primap2.csg._compose.SubstitutionStrategy(),
            )
        ]
    )

    time = pd.date_range("1850-01-01", "2022-01-01", freq="YS")
    anp = np.linspace(0.0, 1.0, len(time))
    bnp = np.linspace(1000.0, 2000.0, len(time))
    anp[0] = np.nan
    anp[1] = np.nan
    bnp[1] = np.nan
    bnp[2] = np.nan
    input_data = xr.DataArray(
        np.array([anp, bnp]).T,
        dims=["time", "source"],
        coords={
            "source": ["A", "B"],
            "time": time,
            "category": "1",
            "area (ISO3)": "MEX",
        },
    )

    result = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )

    enp = np.linspace(0.0, 1.0, len(time))  # same as A
    enp[0] = 1000.0  # filled from B
    enp[1] = np.nan  # not filled, NaN in A and B
    expected = xr.DataArray(
        enp,
        dims=["time"],
        coords={
            "time": time,
            "category": "1",
            "area (ISO3)": "MEX",
        },
    )

    xr.testing.assert_identical(result, expected)
