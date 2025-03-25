#!/usr/bin/env python
"""Tests for _merge.py"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from .utils import assert_aligned_equal, assert_ds_aligned_equal


def test_merge_disjoint_vars(opulent_ds):
    ds_start: xr.Dataset = opulent_ds[["CO2"]]
    ds_merge: xr.Dataset = opulent_ds[["CH4"]]
    ds_result = ds_start.pr.merge(ds_merge)

    assert_ds_aligned_equal(ds_result, opulent_ds[["CH4", "CO2"]])


def test_merge_ds_pass_tolerance(opulent_ds):
    sel = {"time": slice("2000", "2005"), "scenario": "highpop", "source": "RAND2020"}
    ds = opulent_ds.pr.loc[sel]
    ds_start = ds.pr.loc[{"area": ["ARG", "COL"]}]
    ds_merge = ds.pr.loc[{"area": ["ARG", "MEX"]}]

    data_to_modify = ds["CO2"].pr.loc[{"area": ["ARG"]}].pr.sum("area")
    data_to_modify.data *= 1.009

    da_merge = ds["CO2"].pr.set("area", "ARG", data_to_modify, existing="overwrite")
    ds_merge["CO2"] = da_merge
    ds_result = ds_start.pr.merge(ds_merge, tolerance=0.01)

    assert_ds_aligned_equal(ds_result, ds.pr.loc[{"area": ["ARG", "COL", "MEX"]}])


def test_merge_pass_tolerance(opulent_ds):
    # only take part of the countries to have something to actually merge
    da_start = opulent_ds["CO2"].pr.loc[{"area": ["ARG", "COL", "MEX"]}]
    data_to_modify = opulent_ds["CO2"].pr.loc[{"area": ["ARG"]}].pr.sum("area")
    data_to_modify.data *= 1.009
    da_merge = opulent_ds["CO2"].pr.set("area", "ARG", data_to_modify, existing="overwrite")
    da_result = da_start.pr.merge(da_merge, tolerance=0.01)

    assert_aligned_equal(da_result, opulent_ds["CO2"])


def test_merge_fail_tolerance(opulent_ds):
    da_start = opulent_ds["CO2"]
    data_to_modify = opulent_ds["CO2"].pr.loc[{"area": ["ARG"]}].pr.sum("area")
    data_to_modify.data = data_to_modify.data * 1.09
    da_merge = opulent_ds["CO2"].pr.set("area", "ARG", data_to_modify, existing="overwrite")

    with pytest.raises(
        xr.MergeError,
        match="pr.merge error: found discrepancies larger than tolerance",
    ):
        da_start.pr.merge(da_merge, tolerance=0.01)


def test_merge_fail_tolerance_warn(opulent_ds, caplog):
    da_start = opulent_ds["CO2"]
    data_to_modify = opulent_ds["CO2"].pr.loc[{"area": ["ARG"]}].pr.sum("area")
    data_to_modify.data = data_to_modify.data * 1.09
    da_merge = opulent_ds["CO2"].pr.set("area", "ARG", data_to_modify, existing="overwrite")

    da_result = da_start.pr.merge(da_merge, tolerance=0.01, error_on_discrepancy=False)
    assert_aligned_equal(da_result, da_start)
    assert "pr.merge error: found discrepancies larger than tolerance (1.00%) for " in caplog.text


def test_coords_not_matching_ds(opulent_ds):
    ds_start = opulent_ds
    ds_merge = opulent_ds.rename({"time": "year"})

    with pytest.raises(ValueError, match="pr.merge error: coords of objects to merge must agree"):
        ds_start.pr.merge(ds_merge)


def test_coords_not_matching_da(opulent_ds):
    da_start = opulent_ds["CH4"]
    da_merge = opulent_ds["CH4"].rename({"time": "year"})

    with pytest.raises(ValueError, match="pr.merge error: coords of objects to merge must agree"):
        da_start.pr.merge(da_merge)


def test_dims_not_matching_ds(opulent_ds):
    ds_start = opulent_ds
    ds_merge = opulent_ds.rename_dims({"time": "year"})

    with pytest.raises(ValueError, match="pr.merge error: dims of objects to merge must agree"):
        ds_start.pr.merge(ds_merge)


def test_dims_not_matching_da(opulent_ds):
    da_start = opulent_ds["CH4"]
    da_merge = opulent_ds["CH4"].swap_dims({"time": "year"})

    with pytest.raises(ValueError, match="pr.merge error: dims of objects to merge must agree"):
        da_start.pr.merge(da_merge)


# tests to check if additional coords stay in place during merge. Different setups as pr.merge
# uses different methods depending on the setup. as xr.merge is not affected by the coordinate
# bug these tests could also be removed. But if kept they detect if the bug appears for xr.merge
def test_merge_ds_add_coord(opulent_ds):
    country_names = ["Colombia", "Argentina", "Mexico", "Bolivia"]
    ds_full = opulent_ds.assign_coords(country_name=("area (ISO3)", country_names))
    sel = {"time": slice("2000", "2005"), "scenario": "highpop", "source": "RAND2020"}
    ds = ds_full.pr.loc[sel]
    ds_start = ds.pr.loc[{"area": ["ARG", "COL"]}]
    ds_merge = ds.pr.loc[{"area": ["ARG", "MEX"]}]

    ds_result = ds_start.pr.merge(ds_merge, tolerance=0.01)
    ds_compare = ds.pr.loc[{"area (ISO3)": ["ARG", "COL", "MEX"]}]

    # check that the additional coord is present in result
    assert "country_name" in list(ds_result.coords)
    # check that the mapping of country names to country codes is intact
    # (meaning that the additional coordinate is aligned correctly)
    for country in ds_compare.coords["area (ISO3)"].values:
        assert (
            ds_result.coords["country_name"].loc[{"area (ISO3)": country}]
            == ds_compare.coords["country_name"].loc[{"area (ISO3)": country}]
        )


def test_merge_ds_add_coord_tolerance(opulent_ds):
    country_names = ["Colombia", "Argentina", "Mexico", "Bolivia"]
    ds_full = opulent_ds.assign_coords(country_name=("area (ISO3)", country_names))
    sel = {"time": slice("2000", "2005"), "scenario": "highpop", "source": "RAND2020"}
    ds = ds_full.pr.loc[sel]
    ds_start = ds.pr.loc[{"area": ["ARG", "COL"]}]
    ds_merge = ds.pr.loc[{"area": ["ARG", "MEX"]}]

    data_to_modify = ds["CO2"].pr.loc[{"area": ["ARG"]}].pr.sum("area")
    data_to_modify.data *= 1.009

    da_merge = ds["CO2"].pr.set("area", "ARG", data_to_modify, existing="overwrite")
    ds_merge["CO2"] = da_merge
    ds_result = ds_start.pr.merge(ds_merge, tolerance=0.01)
    ds_compare = ds.pr.loc[{"area (ISO3)": ["ARG", "COL", "MEX"]}]

    # check that the additional coord is present in result
    assert "country_name" in list(ds_result.coords)
    # check that the mapping of country names to country codes is intact
    # (meaning that the additional coordinate is aligned correctly)
    for country in ds_compare.coords["area (ISO3)"].values:
        assert (
            ds_result.coords["country_name"].loc[{"area (ISO3)": country}]
            == ds_compare.coords["country_name"].loc[{"area (ISO3)": country}]
        )


def test_merge_ds_add_coord_disjoint_vars(opulent_ds):
    country_names = ["Colombia", "Argentina", "Mexico", "Bolivia"]
    ds_full = opulent_ds.assign_coords(country_name=("area (ISO3)", country_names))

    ds_start: xr.Dataset = ds_full[["CO2"]]
    ds_merge: xr.Dataset = ds_full[["CH4"]]
    ds_result = ds_start.pr.merge(ds_merge)
    ds_compare = ds_full[["CO2", "CH4"]]

    # check that the additional coord is present in result
    assert "country_name" in list(ds_merge.coords)
    # check that the mapping of country names to country codes is intact
    # (meaning that the additional coordinate is aligned correctly)
    for country in ds_compare.coords["area (ISO3)"].values:
        assert (
            ds_result.coords["country_name"].loc[{"area (ISO3)": country}]
            == ds_compare.coords["country_name"].loc[{"area (ISO3)": country}]
        )


def test_merge_message_time_yearly(opulent_ds):
    da_start: xr.DataArray = opulent_ds["CO2"]
    npa = opulent_ds["CO2"].loc[{"time": "2000"}].to_numpy().copy()
    npa[:] = 0.0
    da_merge: xr.DataArray = opulent_ds["CO2"].pr.set(
        "time", "2000", npa.squeeze(), existing="overwrite"
    )
    with pytest.raises(
        xr.MergeError, match=r"found discrepancies larger than tolerance \(1\.00%\) for time=2000,"
    ):
        da_start.pr.merge(da_merge)


def test_merge_message_time_daily(opulent_ds):
    da_start: xr.DataArray = opulent_ds["CO2"]
    da_start = da_start.assign_coords(
        {"time": pd.date_range("2000-01-01", freq="D", periods=len(da_start["time"]))}
    )
    npa = da_start.loc[{"time": "2000-01-02"}].to_numpy().copy()
    npa[:] = 0.0
    da_merge: xr.DataArray = da_start.pr.set(
        "time", "2000-01-02", npa.squeeze(), existing="overwrite"
    )
    with pytest.raises(
        xr.MergeError,
        match=r"found discrepancies larger than tolerance \(1\.00%\) for time=2000-01-02",
    ):
        da_start.pr.merge(da_merge)


def test_log_formatting(minimal_ds, caplog):
    da_start = minimal_ds["CO2"]
    data_to_modify = (
        minimal_ds["CO2"].pr.loc[{"area": ["ARG"], "time": ["2001", "2002"]}].pr.sum("area")
    )
    data_to_modify.data = data_to_modify.data * 1.09
    da_merge = minimal_ds["CO2"].pr.set("area", "ARG", data_to_modify, existing="overwrite")

    da_result = da_start.pr.merge(da_merge, tolerance=0.01, error_on_discrepancy=False)
    assert_aligned_equal(da_result, da_start)
    assert (
        "pr.merge error: found discrepancies larger than tolerance (1.00%) "
        "for area (ISO3)=ARG, source=RAND2020:" in caplog.text
    )
    assert (
        "(CO2)\n             CO2\ntime            \n2001-01-01  0.09"
        "\n2002-01-01  0.09" in caplog.text
    )


def test_log_formatting_single_date(minimal_ds, caplog):
    da_start = minimal_ds["CO2"]
    data_to_modify = minimal_ds["CO2"].pr.loc[{"area": ["ARG"], "time": ["2000"]}].pr.sum("area")
    data_to_modify.data = data_to_modify.data * 1.09
    da_merge = minimal_ds["CO2"].pr.set("area", "ARG", data_to_modify, existing="overwrite")

    da_result = da_start.pr.merge(da_merge, tolerance=0.01, error_on_discrepancy=False)
    assert_aligned_equal(da_result, da_start)
    assert (
        "pr.merge error: found discrepancies larger than tolerance (1.00%) "
        "for time=2000, area (ISO3)=ARG, source=RAND2020:" in caplog.text
    )
    assert "(CO2)\n0.09" in caplog.text


def test_merge_str_encoding(minimal_ds):
    # start data set
    minimal_ds["area (ISO3)"].encoding = {"dtype": np.dtype("<U3")}

    # other data set to merge with
    other_ds = minimal_ds.copy(deep=True)
    new_areas = np.array(["COL", "G20", "G7", "UMBRELLA"], dtype="<U8")
    other_ds = other_ds.assign_coords(
        {"area (ISO3)": xr.DataArray(new_areas, coords={"area (ISO3)": new_areas})}
    )
    other_ds["area (ISO3)"].encoding = {"dtype": np.dtype("<U8")}

    merged_ds = minimal_ds.pr.merge(other_ds)
    assert "dtype" not in merged_ds["area (ISO3)"].encoding
