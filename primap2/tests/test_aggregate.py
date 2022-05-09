#!/usr/bin/env python
"""Tests for _aggregate.py"""

import pathlib

import numpy as np
import pytest
import xarray as xr
import xarray.testing

import primap2
from primap2 import ureg

from . import examples
from .utils import assert_equal


@pytest.fixture(params=["opulent_ds", "opulent_ds[CO2]"])
def opulent_ds_or_da(request):
    """Test with the opulent Dataset or an array taken from it."""
    if request.param == "opulent_ds":
        return examples.opulent_ds()
    elif request.param == "opulent_ds[CO2]":
        return examples.opulent_ds()["CO2"]


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


class TestSum:
    def test_skipna(self):
        coords = [("a", [1, 2]), ("b", [1, 2]), ("c", [1, 2, 3])]
        da = xr.DataArray(
            data=[
                [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
                [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
            ],
            coords=coords,
        )

        b = da.pr.sum(dim="b", skipna_evaluation_dims="a")
        b_expected = xr.DataArray(
            data=[[0, 1, 2], [0, 1, 2]], coords=[coords[0], coords[2]]
        )
        assert np.allclose(b, b_expected, equal_nan=True)

        c = da.pr.sum(dim="b", skipna_evaluation_dims="c")
        c_expected = xr.DataArray(
            data=[[np.nan, 1, 2], [np.nan, 1, 2]], coords=[coords[0], coords[2]]
        )
        assert np.allclose(c, c_expected, equal_nan=True)

        ds = xr.Dataset({"1": da, "2": da.copy()})
        dss = ds.pr.sum(dim="b", skipna_evaluation_dims="a")
        dss_expected = xr.Dataset(
            {
                "1": b_expected,
                "2": b_expected,
            }
        )
        xr.testing.assert_identical(dss, dss_expected)

    def test_inhomogeneous_regression(self, opulent_ds: xr.Dataset):
        ds = opulent_ds
        dss = ds.pr.loc[{"category": ["1", "2", "3", "4", "5"]}].pr.sum("category")
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

    def test_errors(self, opulent_ds_or_da):
        with pytest.raises(
            ValueError,
            match="Only one of 'dim' and 'reduce_to_dim' may be supplied, not both.",
        ):
            opulent_ds_or_da.pr.sum("area", reduce_to_dim="category")

        with pytest.raises(
            ValueError,
            match="Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied",
        ):
            opulent_ds_or_da.pr.sum(
                "area", skipna=True, skipna_evaluation_dims="category"
            )

    def test_error_entity(self, opulent_ds):
        with pytest.raises(
            NotImplementedError,
            match="Summing along the entity dimension is only supported",
        ):
            opulent_ds.pr.sum("entity")

    def test_reduce_to_dim_scalar(self, opulent_ds):
        xr.testing.assert_identical(
            opulent_ds[["CO2", "SF6 (SARGWP100)"]].pr.sum(reduce_to_dim="category"),
            opulent_ds[["CO2", "SF6 (SARGWP100)"]].pr.sum(reduce_to_dim=["category"]),
        )

        xr.testing.assert_identical(
            opulent_ds["CO2"].pr.sum(reduce_to_dim="category"),
            opulent_ds["CO2"].pr.sum(reduce_to_dim=["category"]),
        )

    def test_reduce_to_dim(self, opulent_ds: xr.Dataset):
        xr.testing.assert_identical(
            opulent_ds.pr.sum(reduce_to_dim=["entity", "area"]),
            opulent_ds.sum(set(opulent_ds.dims) - {"area (ISO3)"}, keep_attrs=True),
        )

        shared_ds = opulent_ds[["CO2", "SF6 (SARGWP100)"]]
        xr.testing.assert_allclose(
            shared_ds.pr.sum(reduce_to_dim=["area"]),
            shared_ds.sum(set(shared_ds.dims) - {"area (ISO3)"}, keep_attrs=True)
            .to_array("entity")
            .sum("entity", keep_attrs=True),
        )


def test_any(opulent_ds_or_da):
    xr.testing.assert_identical(
        opulent_ds_or_da.pr.any(dim="area"), opulent_ds_or_da.any(dim="area (ISO3)")
    )


class TestCount:
    def test_inhomogeneous_entity(self, opulent_ds):
        with pytest.raises(NotImplementedError):
            opulent_ds.pr.count("entity")

    def test_entity(self, opulent_ds):
        actual = opulent_ds[["CO2", "SF6"]].pr.count(dim=["entity", "area"])
        assert (actual == 8).all()

    def test_array(self, opulent_ds):
        da = opulent_ds["CO2"]
        actual = da.pr.count(reduce_to_dim=["area"])
        expected = da.count(set(da.dims) - {"area (ISO3)"})
        xr.testing.assert_allclose(actual, expected)

    def test_set(self, opulent_ds):
        ds = opulent_ds
        actual = ds.pr.count(reduce_to_dim=["area", "entity"])
        expected = ds.count(set(ds.dims) - {"area (ISO3)"})
        xr.testing.assert_allclose(actual, expected)


class TestGasBasket:
    # AR4GWP100 values
    sf6 = 22_800
    ch4 = 25

    @pytest.fixture
    def partly_nan_ds(self, empty_ds):
        empty_ds["CO2"][:] = 1 * ureg("Gg CO2 / year")
        empty_ds["SF6"][:] = 1 * ureg("Gg SF6 / year")
        empty_ds["CH4"][:] = 1 * ureg("Gg CH4 / year")
        empty_ds["CH4"].loc[{"area (ISO3)": "COL"}] = np.nan * ureg("Gg CH4 / year")
        return empty_ds

    def test_contents_sum_default(self, partly_nan_ds):
        summed = partly_nan_ds.pr.gas_basket_contents_sum(
            basket="KYOTOGHG (AR4GWP100)",
            basket_contents=["CO2", "SF6", "CH4"],
        )
        expected = partly_nan_ds["KYOTOGHG (AR4GWP100)"].copy()
        expected[:] = (1 + self.sf6 + self.ch4) * ureg("Gg CO2 / year")
        # NaN counted as 0
        expected.loc[{"area (ISO3)": "COL"}] = (1 + self.sf6) * ureg("Gg CO2 / year")
        assert_equal(summed, expected)

    def test_contents_sum_skipna_evaluation_dims(self, partly_nan_ds):
        partly_nan_ds["CH4"].loc[
            {"area (ISO3)": "ARG", "time": "2012"}
        ] = np.nan * ureg("Gg CH4 / year")
        summed = partly_nan_ds.pr.gas_basket_contents_sum(
            basket="KYOTOGHG (AR4GWP100)",
            basket_contents=["CO2", "SF6", "CH4"],
            skipna_evaluation_dims=("time",),
        )
        expected = partly_nan_ds["KYOTOGHG (AR4GWP100)"].copy()
        expected[:] = (1 + self.sf6 + self.ch4) * ureg("Gg CO2 / year")
        # NaN only skipped where all time points NaN
        expected.loc[{"area (ISO3)": "COL"}] = (1 + self.sf6) * ureg("Gg CO2 / year")
        expected.loc[{"area (ISO3)": "ARG", "time": "2012"}] = np.nan * ureg(
            "Gg CO2 / year"
        )
        assert_equal(summed, expected, equal_nan=True)

    def test_contents_sum_skipna(self, partly_nan_ds):
        summed = partly_nan_ds.pr.gas_basket_contents_sum(
            basket="KYOTOGHG (AR4GWP100)",
            basket_contents=["CO2", "SF6", "CH4"],
            skipna=False,
        )
        expected = partly_nan_ds["KYOTOGHG (AR4GWP100)"].copy()
        expected[:] = (1 + self.sf6 + self.ch4) * ureg("Gg CO2 / year")
        # NaNs not skipped
        expected.loc[{"area (ISO3)": "COL"}] = np.nan * ureg("Gg CO2 / year")
        assert_equal(summed, expected, equal_nan=True)

    @pytest.fixture
    def partly_filled_ds(self, partly_nan_ds):
        partly_nan_ds["KYOTOGHG (AR4GWP100)"][:] = 1 * ureg("Gg CO2 / year")
        partly_nan_ds["KYOTOGHG (AR4GWP100)"].loc[
            {"area (ISO3)": "COL"}
        ] = np.nan * ureg("Gg CO2 / year")
        partly_nan_ds["KYOTOGHG (AR4GWP100)"].loc[
            {"area (ISO3)": "BOL", "time": "2020"}
        ] = np.nan * ureg("Gg CO2 / year")
        return partly_nan_ds

    def test_fill_na_from_contents_skipna_evaluation_dims(self, partly_filled_ds):
        filled = partly_filled_ds.pr.fill_na_gas_basket_from_contents(
            basket="KYOTOGHG (AR4GWP100)",
            basket_contents=["CO2", "SF6", "CH4"],
            skipna_evaluation_dims=("time",),
        )
        expected = partly_filled_ds["KYOTOGHG (AR4GWP100)"].copy()
        expected.loc[{"area (ISO3)": "COL"}] = (1 + self.sf6) * ureg("Gg CO2 / year")
        expected.loc[{"area (ISO3)": "BOL", "time": "2020"}] = (
            1 + self.sf6 + self.ch4
        ) * ureg("Gg CO2 / year")
        assert_equal(filled, expected)

    def test_fill_na_from_contents_sel(self, partly_filled_ds):
        filled = partly_filled_ds.pr.fill_na_gas_basket_from_contents(
            basket="KYOTOGHG (AR4GWP100)",
            basket_contents=["CO2", "SF6", "CH4"],
            sel={"area (ISO3)": ["BOL"]},
            skipna_evaluation_dims=("time",),
        )
        expected = partly_filled_ds["KYOTOGHG (AR4GWP100)"].copy()
        expected.loc[{"area (ISO3)": "BOL", "time": "2020"}] = (
            1 + self.sf6 + self.ch4
        ) * ureg("Gg CO2 / year")
        assert_equal(filled, expected, equal_nan=True)

        with pytest.raises(
            ValueError,
            match="The dimension of the selection doesn't match the dimension",
        ):
            partly_filled_ds.pr.fill_na_gas_basket_from_contents(
                basket="KYOTOGHG (AR4GWP100)",
                basket_contents=["CO2", "SF6", "CH4"],
                sel={"area (ISO3)": "BOL"},
                skipna_evaluation_dims=("time",),
            )

    def test_fill_na_from_contents_skipna(self, partly_filled_ds):
        filled = partly_filled_ds.pr.fill_na_gas_basket_from_contents(
            basket="KYOTOGHG (AR4GWP100)",
            basket_contents=["CO2", "SF6", "CH4"],
            skipna=False,
        )
        expected = partly_filled_ds["KYOTOGHG (AR4GWP100)"].copy()
        expected.loc[{"area (ISO3)": "COL"}] = np.nan * ureg("Gg CO2 / year")
        expected.loc[{"area (ISO3)": "BOL", "time": "2020"}] = (
            1 + self.sf6 + self.ch4
        ) * ureg("Gg CO2 / year")
        assert_equal(filled, expected, equal_nan=True)
