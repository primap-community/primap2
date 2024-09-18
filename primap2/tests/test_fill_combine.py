#!/usr/bin/env python
"""Tests for _fill_combine.py

We only test the features regarding the buggy treatment of additional (non-indexed)
coordinates here. All core functionality is assumed to be sufficiently tested in xarray
"""

import numpy as np


def test_fillna_ds_coord_present(minimal_ds):
    # add additional coordinate
    country_names = ["Colombia", "Argentina", "Mexico", "Bolivia"]
    full_ds = minimal_ds.assign_coords(country_name=("area (ISO3)", country_names))

    sel = {"area (ISO3)": ["COL", "MEX"]}
    sel_ds = full_ds.pr.loc[sel]
    nan_ds = full_ds.copy()
    nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] = (
        nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] * np.nan
    )

    result_ds = nan_ds.pr.fillna(sel_ds)

    # assert_ds_aligned_equal(result_ds, full_ds)
    # above fails because data type of country_names differs

    # check that the additional coord is present in result
    assert "country_name" in list(result_ds.coords)
    # check that the mapping of country names to country codes is intact
    # (meaning that the additional coordinate is aligned correctly)
    for country in full_ds.coords["area (ISO3)"].values:
        assert (
            result_ds.coords["country_name"].loc[{"area (ISO3)": country}]
            == full_ds.coords["country_name"].loc[{"area (ISO3)": country}]
        )


def test_fillna_da_coord_present(minimal_ds):
    # add additional coordinate
    country_names = ["Colombia", "Argentina", "Mexico", "Bolivia"]
    full_ds = minimal_ds.assign_coords(country_name=("area (ISO3)", country_names))

    sel = {"area (ISO3)": ["COL", "MEX"]}
    sel_ds = full_ds.pr.loc[sel]
    nan_ds = full_ds.copy()
    nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] = (
        nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] * np.nan
    )

    result_da = nan_ds["CO2"].pr.fillna(sel_ds["CO2"])

    # check that the additional coord is present in result
    assert "country_name" in list(result_da.coords)
    # check that the mapping of country names to country codes is intact
    # (meaning that the additional coordinate is aligned correctly)
    for country in full_ds.coords["area (ISO3)"].values:
        assert (
            result_da.coords["country_name"].loc[{"area (ISO3)": country}]
            == full_ds.coords["country_name"].loc[{"area (ISO3)": country}]
        )


def test_combine_first_ds_coord_present(minimal_ds):
    # add additional coordinate
    country_names = ["Colombia", "Argentina", "Mexico", "Bolivia"]
    full_ds = minimal_ds.assign_coords(country_name=("area (ISO3)", country_names))

    sel = {"area (ISO3)": ["COL", "MEX"]}
    sel_ds = full_ds.pr.loc[sel]
    nan_ds = full_ds.pr.loc[{"area (ISO3)": ["ARG", "COL"]}]
    nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] = (
        nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] * np.nan
    )

    result_ds = nan_ds.pr.combine_first(sel_ds)
    compare_ds = full_ds.pr.loc[{"area (ISO3)": ["ARG", "COL", "MEX"]}]

    # check that the additional coord is present in result
    assert "country_name" in list(result_ds.coords)
    # check that the mapping of country names to country codes is intact
    # (meaning that the additional coordinate is aligned correctly)
    for country in compare_ds.coords["area (ISO3)"].values:
        assert (
            result_ds.coords["country_name"].loc[{"area (ISO3)": country}]
            == compare_ds.coords["country_name"].loc[{"area (ISO3)": country}]
        )


def test_combine_first_da_coord_present(minimal_ds):
    # add additional coordinate
    country_names = ["Colombia", "Argentina", "Mexico", "Bolivia"]
    full_ds = minimal_ds.assign_coords(country_name=("area (ISO3)", country_names))

    sel = {"area (ISO3)": ["COL", "MEX"]}
    sel_ds = full_ds.pr.loc[sel]
    nan_ds = full_ds.pr.loc[{"area (ISO3)": ["ARG", "COL"]}]
    nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] = (
        nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] * np.nan
    )

    result_da = nan_ds["CO2"].pr.combine_first(sel_ds["CO2"])
    compare_da = full_ds["CO2"].pr.loc[{"area (ISO3)": ["ARG", "COL", "MEX"]}]

    # check that the additional coord is present in result
    assert "country_name" in list(result_da.coords)
    # check that the mapping of country names to country codes is intact
    # (meaning that the additional coordinate is aligned correctly)
    for country in compare_da.coords["area (ISO3)"].values:
        assert (
            result_da.coords["country_name"].loc[{"area (ISO3)": country}]
            == compare_da.coords["country_name"].loc[{"area (ISO3)": country}]
        )


# tests to check if xarray bug persists
def test_fillna_ds_xr_fail(minimal_ds):
    # add additional coordinate
    country_names = ["Colombia", "Argentina", "Mexico", "Bolivia"]
    full_ds = minimal_ds.assign_coords(country_name=("area (ISO3)", country_names))

    sel = {"area (ISO3)": ["COL", "MEX"]}
    sel_ds = full_ds.pr.loc[sel]
    nan_ds = full_ds.copy()
    nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] = (
        nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] * np.nan
    )

    result_ds = nan_ds.fillna(sel_ds)

    assert "country_name" not in list(result_ds.coords)


def test_combine_first_ds_xr_fail(minimal_ds):
    # add additional coordinate
    country_names = ["Colombia", "Argentina", "Mexico", "Bolivia"]
    full_ds = minimal_ds.assign_coords(country_name=("area (ISO3)", country_names))

    sel = {"area (ISO3)": ["COL", "MEX"]}
    sel_ds = full_ds.pr.loc[sel]
    nan_ds = full_ds.pr.loc[{"area (ISO3)": ["ARG", "COL"]}]
    nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] = (
        nan_ds["CO2"].pr.loc[{"area (ISO3)": "COL"}] * np.nan
    )

    result_ds = nan_ds.combine_first(sel_ds)

    assert "country_name" not in list(result_ds.coords)
