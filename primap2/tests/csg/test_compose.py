"""Tests for csg/test_compose.py"""
import numpy as np
import pandas as pd
import xarray as xr

import primap2.csg._compose


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
