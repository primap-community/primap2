#!/usr/bin/env python
"""Tests for _downscale.py"""

import numpy as np
import pytest
import xarray as xr

from primap2 import ureg

from .utils import allclose, assert_equal


@pytest.fixture
def gas_downscaling_ds(empty_ds):
    for key in empty_ds:
        empty_ds[key].pint.magnitude[:] = np.nan
    empty_ds["CO2"].loc[{"time": "2002"}] = 1 * ureg("Gg CO2 / year")
    empty_ds["SF6"].loc[{"time": "2002"}] = 1 * ureg("Gg SF6 / year")
    empty_ds["CH4"].loc[{"time": "2002"}] = 1 * ureg("Gg CH4 / year")
    sf6 = 22_800
    ch4 = 25
    empty_ds["KYOTOGHG (AR4GWP100)"][:] = (1 + sf6 + ch4) * ureg("Gg CO2 / year")
    empty_ds["KYOTOGHG (AR4GWP100)"].loc[{"time": "2020"}] = (
        2 * (1 + sf6 + ch4) * ureg("Gg CO2 / year")
    )
    return empty_ds


@pytest.fixture
def dim_downscaling_ds(empty_ds):
    for key in empty_ds:
        empty_ds[key].pint.magnitude[:] = np.nan
    t = empty_ds.loc[{"area (ISO3)": "BOL"}].copy()
    t["area (ISO3)"] = ["CAMB"]  # here, the sum of COL, ARG, MEX, and BOL
    ds = xr.concat([empty_ds, t], dim="area (ISO3)")
    da: xr.DataArray = ds["CO2"]

    da.loc[{"area (ISO3)": ["COL", "ARG", "MEX"], "time": "2002"}] = 1 * ureg("Gg CO2 / year")
    da.loc[{"area (ISO3)": "BOL", "time": "2002"}] = 3 * ureg("Gg CO2 / year")
    da.loc[{"area (ISO3)": "CAMB", "time": "2002"}] = 6 * ureg("Gg CO2 / year")

    da.loc[{"area (ISO3)": ["COL", "ARG", "MEX", "BOL"], "time": "2012"}] = 2 * ureg(
        "Gg CO2 / year"
    )

    da.loc[{"area (ISO3)": "CAMB", "source": "RAND2020"}] = np.concatenate(
        [np.array([6] * 11), np.stack([8, 8]), np.linspace(8, 10, 8)]
    ) * ureg("Gg CO2 / year")
    return ds


@pytest.fixture
def dim_downscaling_da(dim_downscaling_ds):
    return dim_downscaling_ds["CO2"]


@pytest.fixture
def dim_downscaling_expected_da(dim_downscaling_da):
    expected = dim_downscaling_da.copy()

    expected.loc[{"area (ISO3)": ["COL", "ARG", "MEX"], "source": "RAND2020"}] = np.broadcast_to(
        np.concatenate(
            [
                np.array([1, 1]),
                np.linspace(1 / 6, 2 / 8, 11) * np.array([6] * 9 + [8] * 2),
                np.linspace(2, 2 * 10 / 8, 8),
            ]
        ),
        (3, 21),
    ).T * ureg("Gg CO2 / year")
    expected.loc[{"area (ISO3)": "BOL", "source": "RAND2020"}] = np.concatenate(
        [
            np.array([3, 3]),
            np.linspace(3 / 6, 2 / 8, 11) * np.array([6] * 9 + [8] * 2),
            np.linspace(2, 2 * 10 / 8, 8),
        ]
    ) * ureg("Gg CO2 / year")
    return expected


def test_downscale_gas_timeseries(gas_downscaling_ds):
    downscaled = gas_downscaling_ds.pr.downscale_gas_timeseries(
        basket="KYOTOGHG (AR4GWP100)", basket_contents=["CO2", "SF6", "CH4"]
    )
    expected = gas_downscaling_ds.copy()
    expected["CO2"][:] = 1 * ureg("Gg CO2 / year")
    expected["SF6"][:] = 1 * ureg("Gg SF6 / year")
    expected["CH4"][:] = 1 * ureg("Gg CH4 / year")
    expected["CO2"].loc[{"time": "2020"}] = 2 * ureg("Gg CO2 / year")
    expected["SF6"].loc[{"time": "2020"}] = 2 * ureg("Gg SF6 / year")
    expected["CH4"].loc[{"time": "2020"}] = 2 * ureg("Gg CH4 / year")

    xr.testing.assert_identical(downscaled, expected)

    with pytest.raises(
        ValueError,
        match="Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied, not both.",
    ):
        gas_downscaling_ds.pr.downscale_gas_timeseries(
            basket="KYOTOGHG (AR4GWP100)",
            basket_contents=["CO2", "SF6", "CH4"],
            skipna_evaluation_dims=["time"],
            skipna=True,
        )

    gas_downscaling_ds["SF6"].loc[{"time": "2002"}] = 2 * ureg("Gg SF6 / year")

    with pytest.raises(ValueError, match="To continue regardless, set check_consistency=False"):
        gas_downscaling_ds.pr.downscale_gas_timeseries(
            basket="KYOTOGHG (AR4GWP100)", basket_contents=["CO2", "SF6", "CH4"]
        )


def test_downscale_timeseries(dim_downscaling_ds, dim_downscaling_da, dim_downscaling_expected_da):
    downscaled = dim_downscaling_da.pr.downscale_timeseries(
        dim="area (ISO3)", basket="CAMB", basket_contents=["COL", "ARG", "MEX", "BOL"]
    )

    # we need a higher atol, because downscale_timeseries actually does the
    # downscaling using a proper calendar while here we use a calendar where all years
    # have the same length.
    assert_equal(downscaled, dim_downscaling_expected_da, equal_nan=True, atol=0.01)
    allclose(
        downscaled.loc[{"area (ISO3)": "CAMB"}],
        downscaled.loc[{"area (ISO3)": ["COL", "ARG", "MEX", "BOL"]}].sum(dim="area (ISO3)"),
    )

    downscaled_ds = dim_downscaling_ds.pr.downscale_timeseries(
        dim="area (ISO3)", basket="CAMB", basket_contents=["COL", "ARG", "MEX", "BOL"]
    )
    assert_equal(downscaled_ds["CO2"], dim_downscaling_expected_da, equal_nan=True, atol=0.01)

    with pytest.raises(
        ValueError,
        match="Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied, not both.",
    ):
        dim_downscaling_ds.pr.downscale_timeseries(
            dim="area (ISO3)",
            basket="CAMB",
            basket_contents=["COL", "ARG", "MEX", "BOL"],
            skipna_evaluation_dims=["time"],
            skipna=True,
        )

    dim_downscaling_da.loc[{"area (ISO3)": "BOL", "time": "2002"}] = 2 * ureg("Gg CO2 / year")
    with pytest.raises(ValueError, match="To continue regardless, set check_consistency=False"):
        dim_downscaling_da.pr.downscale_timeseries(
            dim="area (ISO3)",
            basket="CAMB",
            basket_contents=["COL", "ARG", "MEX", "BOL"],
        )

    downscaled = dim_downscaling_da.pr.downscale_timeseries(
        dim="area (ISO3)",
        basket="CAMB",
        basket_contents=["COL", "ARG", "MEX", "BOL"],
        check_consistency=False,
    )

    expected = dim_downscaling_da.copy()

    expected.loc[{"area (ISO3)": ["COL", "ARG", "MEX"], "source": "RAND2020"}] = np.broadcast_to(
        np.concatenate(
            [
                np.array([1.2, 1.2, 1]),
                (np.linspace(1 / 5, 2 / 8, 11) * np.array([6] * 9 + [8] * 2))[1:],
                np.linspace(2, 2 * 10 / 8, 8),
            ]
        ),
        (3, 21),
    ).T * ureg("Gg CO2 / year")
    expected.loc[{"area (ISO3)": "BOL", "source": "RAND2020"}] = np.concatenate(
        [
            np.array([2.4, 2.4, 2]),
            (np.linspace(2 / 5, 2 / 8, 11) * np.array([6] * 9 + [8] * 2))[1:],
            np.linspace(2, 2 * 10 / 8, 8),
        ]
    ) * ureg("Gg CO2 / year")

    assert_equal(downscaled, expected, equal_nan=True, atol=0.01)

    downscaled = dim_downscaling_da.pr.downscale_timeseries(
        dim="area (ISO3)",
        basket="CAMB",
        basket_contents=["COL", "ARG", "MEX", "BOL"],
        check_consistency=False,
        sel={"time": slice("2005", "2020")},
    )
    expected = dim_downscaling_da.copy()

    expected.loc[{"area (ISO3)": ["COL", "ARG", "MEX", "BOL"], "source": "RAND2020"}] = (
        np.broadcast_to(
            np.concatenate(
                [
                    np.array(
                        [
                            np.nan,
                            np.nan,
                            1,
                            np.nan,
                            np.nan,
                            6 / 4,
                            6 / 4,
                            6 / 4,
                            6 / 4,
                            6 / 4,
                            6 / 4,
                            2,
                            2,
                        ]
                    ),
                    np.linspace(2, 2 * 10 / 8, 8),
                ]
            ),
            (4, 21),
        ).T
        * ureg("Gg CO2 / year")
    )
    expected.loc[{"area (ISO3)": "BOL", "time": "2002"}] = 2 * ureg("Gg CO2 / year")

    assert_equal(downscaled, expected, equal_nan=True, atol=0.01)


def test_downscale_timeseries_da_zero(dim_downscaling_da, dim_downscaling_expected_da):
    dim_downscaling_da = dim_downscaling_da * 0

    dim_downscaling_expected_da = dim_downscaling_expected_da * 0

    downscaled = dim_downscaling_da.pr.downscale_timeseries(
        dim="area (ISO3)", basket="CAMB", basket_contents=["COL", "ARG", "MEX", "BOL"]
    )

    assert_equal(downscaled, dim_downscaling_expected_da, equal_nan=True)
    allclose(
        downscaled.loc[{"area (ISO3)": "CAMB"}],
        downscaled.loc[{"area (ISO3)": ["COL", "ARG", "MEX", "BOL"]}].sum(dim="area (ISO3)"),
    )


def test_downscale_timeseries_da_partial_zero(dim_downscaling_da, dim_downscaling_expected_da):
    dim_downscaling_da.loc[{"time": "2002"}] = dim_downscaling_da.loc[{"time": "2002"}] * 0

    dim_downscaling_expected_da.loc[
        {"area (ISO3)": ["COL", "ARG", "MEX", "BOL"], "source": "RAND2020"}
    ] = np.broadcast_to(
        np.concatenate(
            [
                np.array([1.5, 1.5]),
                np.array([0]),
                1 / 4 * np.array([6] * 8 + [8] * 2),
                np.linspace(2, 2 * 10 / 8, 8),
            ]
        ),
        (4, 21),
    ).T * ureg("Gg CO2 / year")

    dim_downscaling_expected_da.loc[
        {"area (ISO3)": ["CAMB"], "source": "RAND2020", "time": ["2002"]}
    ] = 0 * ureg("Gg CO2 / year")

    downscaled = dim_downscaling_da.pr.downscale_timeseries(
        dim="area (ISO3)", basket="CAMB", basket_contents=["COL", "ARG", "MEX", "BOL"]
    )

    assert_equal(downscaled, dim_downscaling_expected_da, equal_nan=True)
    allclose(
        downscaled.loc[{"area (ISO3)": "CAMB"}],
        downscaled.loc[{"area (ISO3)": ["COL", "ARG", "MEX", "BOL"]}].sum(dim="area (ISO3)"),
    )


def test_downscale_timeseries_da_zero_sum_error(dim_downscaling_da):
    dim_downscaling_da.loc[{"area (ISO3)": ["COL", "ARG", "MEX", "BOL"]}] = 0 * ureg(
        str(dim_downscaling_da.pint.units)
    )

    with pytest.raises(
        ValueError,
        match=r"pr.downscale_timeseries error: found zero basket content sum for "
        r"non-zero basket data for source=RAND2020:",
    ):
        dim_downscaling_da.pr.downscale_timeseries(
            dim="area (ISO3)",
            basket="CAMB",
            basket_contents=["COL", "ARG", "MEX", "BOL"],
            check_consistency=False,
        )


def test_downscale_gas_timeseries_zero(gas_downscaling_ds):
    for var in gas_downscaling_ds.data_vars:
        gas_downscaling_ds[var].data = gas_downscaling_ds[var].data * 0

    downscaled = gas_downscaling_ds.pr.downscale_gas_timeseries(
        basket="KYOTOGHG (AR4GWP100)", basket_contents=["CO2", "SF6", "CH4"]
    )
    expected = gas_downscaling_ds.copy()
    expected["CO2"][:] = 0 * ureg("Gg CO2 / year")
    expected["SF6"][:] = 0 * ureg("Gg SF6 / year")
    expected["CH4"][:] = 0 * ureg("Gg CH4 / year")

    xr.testing.assert_identical(downscaled, expected)


def test_downscale_gas_timeseries_zero_sum_error(gas_downscaling_ds):
    basket_contents = ["CO2", "SF6", "CH4"]

    for var in basket_contents:
        gas_downscaling_ds[var].loc[{"area (ISO3)": ["COL", "ARG", "MEX", "BOL"]}] = 0 * ureg(
            str(gas_downscaling_ds[var].pint.units)
        )

    with pytest.raises(
        ValueError,
        match=r"pr.downscale_gas_timeseries error: found zero basket content "
        r"sum for non-zero basket data for source=RAND2020:",
    ):
        gas_downscaling_ds.pr.downscale_gas_timeseries(
            basket="KYOTOGHG (AR4GWP100)",
            basket_contents=basket_contents,
            check_consistency=False,
        )


def test_downscale_gas_timeseries_da_partial_zero(gas_downscaling_ds):
    for var in ["CO2", "SF6", "CH4"]:
        gas_downscaling_ds[var].loc[{"time": ["2010"]}] = 0 * gas_downscaling_ds[var].loc[
            {"time": ["2002"]}
        ].squeeze(dim="time")
    gas_downscaling_ds["KYOTOGHG (AR4GWP100)"].loc[{"time": ["2010"]}] = 0 * gas_downscaling_ds[
        "KYOTOGHG (AR4GWP100)"
    ].loc[{"time": ["2002"]}].squeeze(dim="time")

    downscaled = gas_downscaling_ds.pr.downscale_gas_timeseries(
        basket="KYOTOGHG (AR4GWP100)", basket_contents=["CO2", "SF6", "CH4"]
    )

    expected = gas_downscaling_ds.copy()
    expected["CO2"][:] = 1 * ureg("Gg CO2 / year")
    expected["SF6"][:] = 1 * ureg("Gg SF6 / year")
    expected["CH4"][:] = 1 * ureg("Gg CH4 / year")
    expected["CO2"].loc[{"time": "2020"}] = 2 * ureg("Gg CO2 / year")
    expected["SF6"].loc[{"time": "2020"}] = 2 * ureg("Gg SF6 / year")
    expected["CH4"].loc[{"time": "2020"}] = 2 * ureg("Gg CH4 / year")
    expected["CO2"].loc[{"time": "2010"}] = 0 * ureg("Gg CO2 / year")
    expected["SF6"].loc[{"time": "2010"}] = 0 * ureg("Gg SF6 / year")
    expected["CH4"].loc[{"time": "2010"}] = 0 * ureg("Gg CH4 / year")

    xr.testing.assert_identical(downscaled, expected)
