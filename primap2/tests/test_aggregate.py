#!/usr/bin/env python
"""Tests for _aggregate.py"""

import pathlib
import re

import numpy as np
import pytest
import xarray as xr

import primap2
from primap2 import ureg

from . import examples
from .utils import assert_equal


@pytest.fixture(params=["opulent_ds", "opulent_ds[CO2]"])
def opulent_ds_or_da(request):
    """Test with the opulent Dataset or an array taken from it."""
    if request.param == "opulent_ds":
        return examples._cached_opulent_ds.copy(deep=True)
    elif request.param == "opulent_ds[CO2]":
        return examples._cached_opulent_ds["CO2"].copy(deep=True)


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
    def test_skipna_evaluation_dims(self):
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

    def test_skipna(self):
        coords = [("a", [1, 2]), ("b", [1, 2]), ("c", [1, 2, 3])]
        da = xr.DataArray(
            data=[
                [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
                [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
            ],
            coords=coords,
        )

        b0 = da.pr.sum(dim="b", skipna=True, min_count=0)
        b0_expected = xr.DataArray(
            data=[[0, 1, 2], [0, 1, 2]], coords=[coords[0], coords[2]]
        )
        assert np.allclose(b0, b0_expected, equal_nan=True)

        b1 = da.pr.sum(dim="b", skipna=True, min_count=1)
        b1_expected = xr.DataArray(
            data=[[np.nan, 1, 2], [np.nan, 1, 2]], coords=[coords[0], coords[2]]
        )
        assert np.allclose(b1, b1_expected, equal_nan=True)

        b2 = da.pr.sum(dim="b", skipna=True, min_count=2)
        b2_expected = xr.DataArray(
            data=[[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            coords=[coords[0], coords[2]],
        )
        assert np.allclose(b2, b2_expected, equal_nan=True)

        bdef = da.pr.sum(dim="b", skipna=True)
        assert np.allclose(bdef, b1_expected, equal_nan=True)

        ds = xr.Dataset({"1": da, "2": da.copy()})
        dss1 = ds.pr.sum(dim="b", skipna=True, min_count=1)
        dss1_expected = xr.Dataset(
            {
                "1": b1_expected,
                "2": b1_expected,
            }
        )
        xr.testing.assert_identical(dss1, dss1_expected)

        dssdef = ds.pr.sum(dim="b", skipna=True)
        xr.testing.assert_identical(dssdef, dss1_expected)

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
            atol=0,
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

    # AR6GWP100 values
    sf6_ar6 = 25_200
    ch4_ar6 = 27.9

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
        partly_nan_ds["CH4"].loc[{"area (ISO3)": "ARG", "time": "2012"}] = (
            np.nan * ureg("Gg CH4 / year")
        )
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

    def test_contents_sum_new_entity(self, partly_nan_ds):
        summed = partly_nan_ds.pr.gas_basket_contents_sum(
            basket="KYOTOGHG (AR6GWP100)",
            basket_contents=["CO2", "SF6", "CH4"],
        )
        expected = partly_nan_ds["KYOTOGHG (AR4GWP100)"].copy()
        expected[:] = (1 + self.sf6_ar6 + self.ch4_ar6) * ureg("Gg CO2 / year")
        # NaN counted as 0
        expected.loc[{"area (ISO3)": "COL"}] = (1 + self.sf6_ar6) * ureg(
            "Gg CO2 / year"
        )
        expected.name = "KYOTOGHG (AR6GWP100)"
        expected.attrs = {"gwp_context": "AR6GWP100", "entity": "KYOTOGHG"}
        assert_equal(summed, expected)

    @pytest.fixture
    def partly_filled_ds(self, partly_nan_ds):
        partly_nan_ds["KYOTOGHG (AR4GWP100)"][:] = 1 * ureg("Gg CO2 / year")
        partly_nan_ds["KYOTOGHG (AR4GWP100)"].loc[{"area (ISO3)": "COL"}] = (
            np.nan * ureg("Gg CO2 / year")
        )
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

    def test_add_aggregates_variables_no_input(self, partly_filled_ds, caplog):
        """
        test if no input data handled correctly by add_aggregates_variables
        """
        filled = partly_filled_ds.pr.add_aggregates_variables(
            gas_baskets={
                "FGASES (SARGWP100)": ["SF6 (SARGWP100)"],
            },
        )

        assert "FGASES (SARGWP100) not created. No input data present." in caplog.text
        assert "FGASES (SARGWP100)" not in filled.data_vars

    def test_add_aggregates_variables_partial_input(self, partly_filled_ds, caplog):
        """
        test if partial input data handled correctly by add_aggregates_variables
        """
        filled = partly_filled_ds.pr.add_aggregates_variables(
            gas_baskets={
                "FGASES (SARGWP100)": ["SF6", "HFC32"],
            },
        )

        assert (
            "Not all variables present for FGASES (SARGWP100). Missing: " "['HFC32']"
        ) in caplog.text
        assert "FGASES (SARGWP100)" in filled.data_vars
        xr.testing.assert_allclose(
            filled["FGASES (SARGWP100)"],
            filled["SF6"].pr.convert_to_gwp("SARGWP100", "Gg CO2 / a"),
        )
        filled.pr.ensure_valid()

    def test_add_aggregates_variables_tolerance(self, partly_filled_ds, caplog):
        """
        test if the tolerance setting works (the data in partly_filled_ds is
        already inconsistent, so we can use it for the test directly)
        """
        with pytest.raises(
            xr.MergeError,
            match="pr.merge error: found discrepancies " "larger than tolerance",
        ):
            partly_filled_ds.pr.add_aggregates_variables(
                gas_baskets={
                    "KYOTOGHG (AR4GWP100)": ["CO2", "SF6", "CH4"],
                },
            )

        filled = partly_filled_ds.pr.add_aggregates_variables(
            gas_baskets={
                "KYOTOGHG (AR4GWP100)": ["CO2", "SF6", "CH4"],
            },
            tolerance=30000,
        )
        assert "KYOTOGHG (AR4GWP100)" in filled.data_vars
        filled.pr.ensure_valid()

        filled = partly_filled_ds.pr.add_aggregates_variables(
            gas_baskets={
                "KYOTOGHG (AR4GWP100)": {
                    "sources": ["CO2", "SF6", "CH4"],
                    "tolerance": 30000,
                },
            }
        )
        assert "KYOTOGHG (AR4GWP100)" in filled.data_vars

    def test_add_aggregates_variables_filter(self, partly_filled_ds, caplog):
        """
        test if the filter setting works
        """
        reference_ds = partly_filled_ds.pr.add_aggregates_variables(
            gas_baskets={
                "test (SARGWP100)": ["CO2", "SF6", "CH4"],
            },
        )

        filtered_ds = partly_filled_ds.pr.add_aggregates_variables(
            gas_baskets={
                "test (SARGWP100)": {
                    "sources": ["CO2", "SF6", "CH4"],
                    "filter": {"area (ISO3)": ["COL"]},
                },
            },
        )

        assert (
            reference_ds["test (SARGWP100)"].pr.loc[{"area (ISO3)": ["COL"]}]
            == filtered_ds["test (SARGWP100)"].pr.loc[{"area (ISO3)": ["COL"]}]
        ).all()
        assert (
            filtered_ds["test (SARGWP100)"]
            .pr.loc[{"area (ISO3)": ["BOL"]}]
            .isnull()
            .all()
        )

    def test_add_aggregates_variables_error_basket_type(self, partly_filled_ds, caplog):
        """
        test error on unrecognized aggregation definition
        """
        with pytest.raises(
            ValueError,
            match=re.escape("Unrecognized basket type for 'test (SARGWP100)'"),
        ):
            partly_filled_ds.pr.add_aggregates_variables(
                gas_baskets={
                    "test (SARGWP100)": "CO2",
                },
            )


class TestAddAggregatesCoordinates:
    """
    Tests for the add_aggregates_categories method.

    No tests for skipna etc as that is just passed on to sum
    """

    def test_add_aggregates_coordinates_tolerance(self, minimal_ds):
        """
        test if the tolerance setting works
        """

        test_ds = minimal_ds.pr.add_aggregates_coordinates(
            agg_info={
                "area (ISO3)": {
                    "all": {
                        "sources": ["COL", "ARG", "MEX", "BOL"],
                    }
                }
            }
        )

        with pytest.raises(
            xr.MergeError,
            match="pr.merge error: found discrepancies " "larger than tolerance",
        ):
            test_ds.pr.add_aggregates_coordinates(
                agg_info={
                    "area (ISO3)": {
                        "all": {
                            "sources": ["COL", "ARG"],
                        }
                    }
                }
            )

        test_ds.pr.add_aggregates_coordinates(
            agg_info={
                "area (ISO3)": {
                    "all": {"sources": ["COL", "ARG", "MEX", "BOL"], "tolerance": 5}
                }
            }
        )

    def test_add_aggregates_coordinates_result(self, minimal_ds):
        """
        test if aggregated value timeseries are present and correct
        """
        # with the complex configuration
        test_ds = minimal_ds.pr.add_aggregates_coordinates(
            agg_info={
                "area (ISO3)": {
                    "all": {
                        "sources": ["COL", "ARG", "MEX", "BOL"],
                    }
                }
            }
        )

        expected_result = minimal_ds["CO2"].pr.sum(dim="area (ISO3)")
        actual_result = (
            test_ds["CO2"].pr.loc[{"area (ISO3)": ["all"]}].pr.sum(dim="area (ISO3)")
        )
        xr.testing.assert_allclose(expected_result, actual_result)

        # with the simplified configuration
        test_ds = minimal_ds.pr.add_aggregates_coordinates(
            agg_info={
                "area (ISO3)": {
                    "all": ["COL", "ARG", "MEX", "BOL"],
                }
            }
        )

        actual_result = (
            test_ds["CO2"].pr.loc[{"area (ISO3)": ["all"]}].pr.sum(dim="area (ISO3)")
        )
        xr.testing.assert_allclose(expected_result, actual_result)

    def test_add_aggregates_coordinates_result_filter(self, minimal_ds):
        """
        test if filtering works and only the selected time series are actually
        computed and correct
        """
        # filter entity
        test_ds = minimal_ds.pr.add_aggregates_coordinates(
            agg_info={
                "area (ISO3)": {
                    "all": {
                        "sources": ["COL", "ARG", "MEX", "BOL"],
                        "filter": {"entity": ["SF6"]},
                    }
                }
            }
        )

        # as we filter for entity we expect the variables SF6 and SF6 (SARGWP100)
        # to be aggregated but for CO2 we expect np.nan
        expected_result_SF6 = minimal_ds["SF6"].pr.sum(dim="area (ISO3)")
        actual_result_SF6 = (
            test_ds["SF6"].pr.loc[{"area (ISO3)": ["all"]}].pr.sum(dim="area (ISO3)")
        )
        xr.testing.assert_allclose(expected_result_SF6, actual_result_SF6)
        expected_result_SF6GWP = minimal_ds["SF6 (SARGWP100)"].pr.sum(dim="area (ISO3)")
        actual_result_SF6GWP = (
            test_ds["SF6 (SARGWP100)"]
            .pr.loc[{"area (ISO3)": ["all"]}]
            .pr.sum(dim="area (ISO3)")
        )
        xr.testing.assert_allclose(expected_result_SF6GWP, actual_result_SF6GWP)

        expected_result_CO2 = xr.full_like(expected_result_SF6, np.nan).pr.quantify(
            units="Gg CO2 / year"
        )
        actual_result_CO2 = (
            test_ds["CO2"]
            .pr.loc[{"area (ISO3)": ["all"]}]
            .pr.sum(dim="area (ISO3)", skipna=True, min_count=1)
        )
        xr.testing.assert_allclose(expected_result_CO2, actual_result_CO2)

        # filter variable
        test_ds = minimal_ds.pr.add_aggregates_coordinates(
            agg_info={
                "area (ISO3)": {
                    "all": {
                        "sources": ["COL", "ARG", "MEX", "BOL"],
                        "filter": {"variable": ["SF6"]},
                    }
                }
            }
        )

        # as we filter for entity we expect the variables SF6 and SF6 (SARGWP100)
        # to be aggregated but for CO2 we expect np.nan
        expected_result_SF6 = minimal_ds["SF6"].pr.sum(dim="area (ISO3)")
        actual_result_SF6 = (
            test_ds["SF6"].pr.loc[{"area (ISO3)": ["all"]}].pr.sum(dim="area (ISO3)")
        )
        xr.testing.assert_allclose(expected_result_SF6, actual_result_SF6)

        expected_result_SF6GWP = xr.full_like(expected_result_SF6, np.nan).pr.quantify(
            units="Gg CO2 / year"
        )
        # expected_result_SF6GWP = expected_result_SF6GWP.pr.convert_to_gwp
        actual_result_SF6GWP = (
            test_ds["SF6 (SARGWP100)"]
            .pr.loc[{"area (ISO3)": ["all"]}]
            .pr.sum(dim="area (ISO3)", skipna=True, min_count=1)
        )
        xr.testing.assert_allclose(expected_result_SF6GWP, actual_result_SF6GWP)

        expected_result_CO2 = xr.full_like(expected_result_SF6, np.nan).pr.quantify(
            units="Gg CO2 / year"
        )
        actual_result_CO2 = (
            test_ds["CO2"]
            .pr.loc[{"area (ISO3)": ["all"]}]
            .pr.sum(dim="area (ISO3)", skipna=True, min_count=1)
        )
        xr.testing.assert_allclose(expected_result_CO2, actual_result_CO2)

    def test_add_aggregates_coordinates_warning(self, minimal_ds, caplog):
        """
        Test warnings
        """
        # test warning for missing input
        minimal_ds.pr.add_aggregates_coordinates(
            agg_info={
                "area": {
                    "all": {
                        "sources": ["COL", "ARG", "MEX", "BOL", "DEU"],
                    }
                }
            }
        )
        assert (
            "Not all source values present for 'all'"
            " in coordinate 'area (ISO3)'. "
            "Missing: {'DEU'}"
        ) in caplog.text

        # test warning for fully missing input
        minimal_ds.pr.add_aggregates_coordinates(
            agg_info={
                "area (ISO3)": {
                    "all": {
                        "sources": ["DEU"],
                    }
                }
            }
        )
        assert (
            "No source value present for 'all' in "
            "coordinate 'area (ISO3)'. Missing: {'DEU'}."
        ) in caplog.text

        # test all nan warning
        test_ds = minimal_ds.copy(deep=True)
        test_ds["CO2"] = xr.full_like(test_ds["CO2"], np.nan).pr.quantify(
            units="Gg CO2 / year"
        )
        test_ds.pr.add_aggregates_coordinates(
            agg_info={
                "area (ISO3)": {
                    "all": {
                        "sources": ["COL", "ARG", "MEX", "BOL"],
                    }
                }
            }
        )
        assert (
            "All input data nan for 'all' in " "coordinate 'area (ISO3)'."
        ) in caplog.text

    def test_add_aggregates_coordinates_errors(self, minimal_ds):
        """
        test errors
        """
        # additional coordinate not present
        with pytest.raises(
            ValueError,
            match="Additional coordinate 'area_name' specified but not "
            "present in data",
        ):
            minimal_ds.pr.add_aggregates_coordinates(
                agg_info={
                    "area (ISO3)": {
                        "all": {
                            "sources": ["COL", "ARG", "MEX", "BOL"],
                            "area_name": "All Countries",
                        }
                    }
                }
            )

        # unrecognized aggregation definition
        with pytest.raises(
            ValueError, match="Unrecognized aggregation definition for 'all'"
        ):
            minimal_ds.pr.add_aggregates_coordinates(
                agg_info={"area (ISO3)": {"all": "COL"}}
            )

    def test_add_aggregates_coordinates_add_coord(self, minimal_ds):
        """test error on additional coordinate"""
        test_ds = minimal_ds.copy(deep=True)
        area_name = ["COL", "ARG", "MEX", "BOL"]
        test_ds = test_ds.assign_coords(area_name=("area (ISO3)", area_name))

        test_ds = test_ds.pr.add_aggregates_coordinates(
            agg_info={
                "area (ISO3)": {
                    "all": {
                        "sources": ["COL", "ARG", "MEX", "BOL"],
                        "area_name": "All Countries",
                    }
                }
            }
        )

        assert "All Countries" in test_ds.coords["area_name"]


# filter, also for individual items in the list and check if results fine
# filter for entity, variable and a coordinate
