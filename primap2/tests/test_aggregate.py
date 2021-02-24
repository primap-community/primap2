#!/usr/bin/env python
"""Tests for _aggregate.py"""

import pathlib

import numpy as np
import pytest
import xarray as xr
import xarray.testing

import primap2
from primap2 import ureg

from .utils import assert_equal


def test_fill_all_na():
    coords = [("a", [1, 2]), ("b", [1, 2]), ("c", [1, 2, 3])]
    da = xr.DataArray(
        data=[
            [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
            [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
        ],
        coords=coords,
    )
    a = da.pr.fill_all_na(dim="b", value=0)
    a_expected = xr.DataArray(
        data=[
            [[0, np.nan, np.nan], [0, 1, 2]],
            [[0, np.nan, np.nan], [0, 1, 2]],
        ],
        coords=coords,
    )
    assert np.allclose(a, a_expected, equal_nan=True)

    b = da.pr.fill_all_na(dim=("a", "b"), value=0)
    assert np.allclose(b, a_expected, equal_nan=True)

    c = da.pr.fill_all_na(dim=("b", "c"), value=0)
    assert np.allclose(c, da, equal_nan=True)

    d = da.pr.fill_all_na(dim="c", value=0)
    d_expected = xr.DataArray(
        data=[
            [[0, 0, 0], [np.nan, 1, 2]],
            [[0, 0, 0], [np.nan, 1, 2]],
        ],
        coords=coords,
    )
    assert np.allclose(d, d_expected, equal_nan=True)

    e = da.pr.fill_all_na(dim=[], value=0)
    assert np.allclose(e, da, equal_nan=True)

    ds = xr.Dataset({"1": da, "2": da.copy()})
    dsf = ds.pr.fill_all_na(dim="b", value=0)
    assert np.allclose(dsf["1"], a_expected, equal_nan=True)
    assert np.allclose(dsf["2"], a_expected, equal_nan=True)


def test_sum_skip_allna():
    coords = [("a", [1, 2]), ("b", [1, 2]), ("c", [1, 2, 3])]
    da = xr.DataArray(
        data=[
            [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
            [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
        ],
        coords=coords,
    )

    a = da.pr.sum_skip_all_na(dim="a")
    a_expected = xr.DataArray(
        data=[
            [np.nan, np.nan, np.nan],
            [np.nan, 2, 4],
        ],
        coords=coords[1:],
    )
    assert np.allclose(a, a_expected, equal_nan=True)

    b = da.pr.sum_skip_all_na(dim="b", skipna_evaluation_dims="a")
    b_expected = xr.DataArray(
        data=[[0, 1, 2], [0, 1, 2]], coords=[coords[0], coords[2]]
    )
    assert np.allclose(b, b_expected, equal_nan=True)

    c = da.pr.sum_skip_all_na(dim="b", skipna_evaluation_dims="c")
    c_expected = xr.DataArray(
        data=[[np.nan, 1, 2], [np.nan, 1, 2]], coords=[coords[0], coords[2]]
    )
    assert np.allclose(c, c_expected, equal_nan=True)

    ds = xr.Dataset({"1": da, "2": da.copy()})
    dss = ds.pr.sum_skip_all_na(dim="a")
    dss_expected = xr.Dataset(
        {
            "1": xr.DataArray(
                data=[
                    [np.nan, np.nan, np.nan],
                    [np.nan, 2, 4],
                ],
                coords=coords[1:],
            ),
            "2": xr.DataArray(
                data=[
                    [np.nan, np.nan, np.nan],
                    [np.nan, 2, 4],
                ],
                coords=coords[1:],
            ),
        }
    )
    xr.testing.assert_identical(dss, dss_expected)


def test_sum_skip_allna_inhomogeneous(opulent_ds: xr.Dataset):
    ds = opulent_ds
    dss = ds.pr.loc[{"category": ["1", "2", "3", "4", "5"]}].pr.sum_skip_all_na(
        "category"
    )
    xr.testing.assert_identical(dss["population"], ds["population"])
    actual = dss["CO2"]
    expected = (
        ds["CO2"]
        .pr.loc[{"category": ["1", "2", "3", "4", "5"]}]
        .sum("category (IPCC 2006)", keep_attrs=True)
    )
    assert_equal(actual, expected)

    # uncomment to refresh this regression test
    # dss.pr.to_netcdf(
    #    pathlib.Path(__file__).parent
    #    / "data"
    #    / "test_sum_skip_allna_inhomogeneous_result.nc"
    # )

    actual = dss
    expected = primap2.open_dataset(
        pathlib.Path(__file__).parent
        / "data"
        / "test_sum_skip_allna_inhomogeneous_result.nc"
    )
    xr.testing.assert_identical(actual, expected)


def test_gas_basket_contents_sum(empty_ds):
    empty_ds["CO2"][:] = 1 * ureg("Gg CO2 / year")
    empty_ds["SF6"][:] = 1 * ureg("Gg SF6 / year")
    empty_ds["CH4"][:] = 1 * ureg("Gg CH4 / year")
    empty_ds["CH4"].loc[{"area (ISO3)": "COL"}] = np.nan * ureg("Gg CH4 / year")

    summed = empty_ds.pr.gas_basket_contents_sum(
        basket="KYOTOGHG (AR4GWP100)",
        basket_contents=["CO2", "SF6", "CH4"],
        skipna_evaluation_dims=("time",),
    )
    expected = empty_ds["KYOTOGHG (AR4GWP100)"].copy()
    sf6 = 22_800
    ch4 = 25
    expected[:] = (1 + sf6 + ch4) * ureg("Gg CO2 / year")
    expected.loc[{"area (ISO3)": "COL"}] = (1 + sf6) * ureg("Gg CO2 / year")
    assert_equal(summed, expected)

    summed = empty_ds.pr.gas_basket_contents_sum(
        basket="KYOTOGHG (AR4GWP100)",
        basket_contents=["CO2", "SF6", "CH4"],
    )
    expected = empty_ds["KYOTOGHG (AR4GWP100)"].copy()
    expected[:] = (1 + sf6 + ch4) * ureg("Gg CO2 / year")
    expected.loc[{"area (ISO3)": "COL"}] = np.nan * ureg("Gg CO2 / year")
    assert_equal(summed, expected, equal_nan=True)


def test_fill_na_gas_basket_from_contents(empty_ds):
    empty_ds["CO2"][:] = 1 * ureg("Gg CO2 / year")
    empty_ds["SF6"][:] = 1 * ureg("Gg SF6 / year")
    empty_ds["CH4"][:] = 1 * ureg("Gg CH4 / year")
    empty_ds["CH4"].loc[{"area (ISO3)": "COL"}] = np.nan * ureg("Gg CH4 / year")
    empty_ds["KYOTOGHG (AR4GWP100)"][:] = 1 * ureg("Gg CO2 / year")
    empty_ds["KYOTOGHG (AR4GWP100)"].loc[{"area (ISO3)": "COL"}] = np.nan * ureg(
        "Gg CO2 / year"
    )
    empty_ds["KYOTOGHG (AR4GWP100)"].loc[
        {"area (ISO3)": "BOL", "time": "2020"}
    ] = np.nan * ureg("Gg CO2 / year")

    filled = empty_ds.pr.fill_na_gas_basket_from_contents(
        basket="KYOTOGHG (AR4GWP100)",
        basket_contents=["CO2", "SF6", "CH4"],
        skipna_evaluation_dims=("time",),
    )
    expected = empty_ds["KYOTOGHG (AR4GWP100)"].copy()
    sf6 = 22_800
    ch4 = 25
    expected.loc[{"area (ISO3)": "COL"}] = (1 + sf6) * ureg("Gg CO2 / year")
    expected.loc[{"area (ISO3)": "BOL", "time": "2020"}] = (1 + sf6 + ch4) * ureg(
        "Gg CO2 / year"
    )
    assert_equal(filled, expected)

    filled = empty_ds.pr.fill_na_gas_basket_from_contents(
        basket="KYOTOGHG (AR4GWP100)",
        basket_contents=["CO2", "SF6", "CH4"],
        sel={"area (ISO3)": ["BOL"]},
        skipna_evaluation_dims=("time",),
    )
    expected = empty_ds["KYOTOGHG (AR4GWP100)"].copy()
    expected.loc[{"area (ISO3)": "BOL", "time": "2020"}] = (1 + sf6 + ch4) * ureg(
        "Gg CO2 / year"
    )
    assert_equal(filled, expected, equal_nan=True)

    with pytest.raises(
        ValueError, match="The dimension of the selection doesn't match the dimension"
    ):
        empty_ds.pr.fill_na_gas_basket_from_contents(
            basket="KYOTOGHG (AR4GWP100)",
            basket_contents=["CO2", "SF6", "CH4"],
            sel={"area (ISO3)": "BOL"},
            skipna_evaluation_dims=("time",),
        )
