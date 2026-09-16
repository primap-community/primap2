"""Tests that all functions which sum data handle skipna consistently."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2  # noqa: F401
from primap2 import ureg

from .utils import assert_equal


@pytest.fixture
def na_ds() -> xr.Dataset:
    """Data with an all-NA time series (ARG) and a partly NA time series (MEX).

    The area CAMB is the sum of COL, ARG, and MEX, and the gas basket KYOTOGHG is the
    sum of CO2 and CH4.
    """
    time = pd.date_range("2000-01-01", "2003-01-01", freq="YS")
    coords = {"area (ISO3)": ["COL", "ARG", "MEX", "CAMB"], "time": time}
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [np.nan, np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0, np.nan],
            [2.0, 4.0, 6.0, 8.0],
        ]
    )
    ds = xr.Dataset(
        {
            "CO2": xr.DataArray(
                values, coords=coords, attrs={"entity": "CO2", "units": "Gg CO2 / year"}
            ),
            "CH4": xr.DataArray(
                values, coords=coords, attrs={"entity": "CH4", "units": "Gg CH4 / year"}
            ),
            "KYOTOGHG (AR4GWP100)": xr.DataArray(
                values * 26,
                coords=coords,
                attrs={
                    "entity": "KYOTOGHG",
                    "gwp_context": "AR4GWP100",
                    "units": "Gg CO2 / year",
                },
            ),
        },
        attrs={"area": "area (ISO3)"},
    ).pr.quantify()
    return ds


BASKET = "KYOTOGHG (AR4GWP100)"
AREAS = ["COL", "ARG", "MEX"]


def da_sum(ds, **kwargs):
    return ds["CO2"].pr.sum(dim="area", **kwargs)


def ds_sum(ds, **kwargs):
    return ds.pr.sum(dim="area", **kwargs)


def gas_basket_contents_sum(ds, **kwargs):
    return ds.pr.gas_basket_contents_sum(basket=BASKET, basket_contents=["CO2", "CH4"], **kwargs)


def fill_na_gas_basket_from_contents(ds, **kwargs):
    return ds.pr.fill_na_gas_basket_from_contents(
        basket=BASKET, basket_contents=["CO2", "CH4"], **kwargs
    )


def da_add_aggregates_coordinates(ds, **kwargs):
    return ds["CO2"].pr.add_aggregates_coordinates(
        agg_info={"area (ISO3)": {"all": AREAS}}, **kwargs
    )


def ds_add_aggregates_coordinates(ds, **kwargs):
    return ds.pr.add_aggregates_coordinates(agg_info={"area (ISO3)": {"all": AREAS}}, **kwargs)


def add_aggregates_variables(ds, **kwargs):
    return ds.pr.add_aggregates_variables(
        gas_baskets={"KYOTOGHG (AR6GWP100)": ["CO2", "CH4"]}, **kwargs
    )


def da_downscale_timeseries(ds, **kwargs):
    return ds["CO2"].pr.downscale_timeseries(
        dim="area (ISO3)",
        basket="CAMB",
        basket_contents=AREAS,
        check_consistency=False,
        **kwargs,
    )


def ds_downscale_timeseries(ds, **kwargs):
    return ds.pr.downscale_timeseries(
        dim="area (ISO3)",
        basket="CAMB",
        basket_contents=AREAS,
        check_consistency=False,
        **kwargs,
    )


def downscale_gas_timeseries(ds, **kwargs):
    return ds.pr.downscale_gas_timeseries(
        basket=BASKET,
        basket_contents=["CO2", "CH4"],
        check_consistency=False,
        **kwargs,
    )


SUMMING_FUNCTIONS = [
    da_sum,
    ds_sum,
    gas_basket_contents_sum,
    fill_na_gas_basket_from_contents,
    da_add_aggregates_coordinates,
    ds_add_aggregates_coordinates,
    add_aggregates_variables,
]
DOWNSCALING_FUNCTIONS = [
    da_downscale_timeseries,
    ds_downscale_timeseries,
    downscale_gas_timeseries,
]


@pytest.mark.parametrize("func", SUMMING_FUNCTIONS + DOWNSCALING_FUNCTIONS)
class TestConsistentArguments:
    """All functions have to accept the same combinations of arguments."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"skipna": True},
            {"skipna": False},
            {"skipna": None},
            {"skipna_evaluation_dims": "time"},
            {"skipna_evaluation_dims": "time", "skipna": None},
            {"min_count": 1},
            {"skipna_evaluation_dims": "time", "min_count": 1},
        ],
    )
    def test_accepted(self, func, na_ds, kwargs):
        func(na_ds, **kwargs)

    def test_skipna_true_and_evaluation_dims(self, func, na_ds):
        with pytest.raises(
            ValueError,
            match="Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied",
        ):
            func(na_ds, skipna=True, skipna_evaluation_dims="time")

    def test_skipna_false_and_evaluation_dims(self, func, na_ds):
        if func in DOWNSCALING_FUNCTIONS:
            pytest.skip("deprecated, but still supported for downscaling functions")
        with pytest.raises(
            ValueError,
            match="Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied",
        ):
            func(na_ds, skipna=False, skipna_evaluation_dims="time")

    def test_default_equals_skipna_true(self, func, na_ds):
        xr.testing.assert_identical(func(na_ds), func(na_ds, skipna=True))


@pytest.mark.parametrize("func", DOWNSCALING_FUNCTIONS)
def test_downscaling_deprecated_skipna_false_and_evaluation_dims(func, na_ds):
    with pytest.warns(DeprecationWarning, match="skipna_evaluation_dims alone"):
        deprecated = func(na_ds, skipna=False, skipna_evaluation_dims="time")
    xr.testing.assert_identical(deprecated, func(na_ds, skipna_evaluation_dims="time"))


@pytest.mark.parametrize("func", [da_add_aggregates_coordinates, ds_add_aggregates_coordinates])
def test_add_aggregates_coordinates_skipna_evaluation_dims(func, na_ds):
    result = func(na_ds, skipna_evaluation_dims="time")
    if isinstance(result, xr.Dataset):
        result = result["CO2"]
    actual = result.pr.loc[{"area": "all"}]
    # ARG is skipped because it is NA for all points in time, but MEX is not skipped
    expected = [2.0, 4.0, 6.0, np.nan] * ureg("Gg CO2 / year")
    np.testing.assert_allclose(actual.pint.magnitude, expected.magnitude)


def test_add_aggregates_variables_skipna_evaluation_dims(na_ds):
    result = add_aggregates_variables(na_ds, skipna_evaluation_dims="time")
    expected = na_ds.pr.gas_basket_contents_sum(
        basket="KYOTOGHG (AR6GWP100)",
        basket_contents=["CO2", "CH4"],
        skipna_evaluation_dims="time",
    )
    assert_equal(result["KYOTOGHG (AR6GWP100)"], expected, equal_nan=True)
    # MEX is not skipped in 2003 because not all points in time are NA
    assert np.isnan(expected.pr.loc[{"area": "MEX", "time": "2003"}].pint.magnitude)
    assert not np.isnan(expected.pr.loc[{"area": "ARG"}].pint.magnitude).any()


def test_downscale_timeseries_min_count(na_ds):
    # with min_count=0, the all-NA contents of 2003 are summed to zero, which is
    # inconsistent with the non-zero basket
    na_ds["CO2"].loc[{"area (ISO3)": "COL", "time": "2003"}] = np.nan * ureg("Gg CO2 / year")
    da_downscale_timeseries(na_ds)
    with pytest.raises(ValueError, match="found zero basket content sum"):
        da_downscale_timeseries(na_ds, min_count=0)
