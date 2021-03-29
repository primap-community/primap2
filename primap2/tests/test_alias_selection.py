"""Tests for _alias_selection.py"""

import pytest
import xarray as xr
import xarray.testing


@pytest.mark.parametrize(
    ["alias", "full_name"],
    [
        ("time", "time"),
        ("area", "area (ISO3)"),
        ("category", "category (IPCC 2006)"),
        ("cat", "category (IPCC 2006)"),
        ("animal", "animal (FAOSTAT)"),
        ("product", "product (FAOSTAT)"),
        ("scenario", "scenario (FAOSTAT)"),
        ("scen", "scenario (FAOSTAT)"),
        ("provenance", "provenance"),
        ("model", "model"),
        ("source", "source"),
        ("CO2", "CO2"),
        ("population", "population"),
    ],
)
def test_pr_getitem(opulent_ds, alias, full_name):
    da = opulent_ds.pr[alias]
    assert da.name == full_name


@pytest.mark.parametrize(
    ["alias", "full_name"],
    [
        ("time", "time"),
        ("area", "area (ISO3)"),
        ("category", "category (IPCC 2006)"),
        ("cat", "category (IPCC 2006)"),
        ("animal", "animal (FAOSTAT)"),
        ("product", "product (FAOSTAT)"),
        ("scenario", "scenario (FAOSTAT)"),
        ("scen", "scenario (FAOSTAT)"),
        ("provenance", "provenance"),
        ("model", "model"),
        ("source", "source"),
        ("CO2", "CO2"),
        ("population", "population"),
    ],
)
def test_pr_getitem_no_attrs(opulent_ds, alias, full_name):
    da = opulent_ds.notnull().pr[alias]
    assert da.name == full_name


@pytest.mark.parametrize(
    ["alias", "full_name"],
    [
        ("time", "time"),
        ("area", "area (ISO3)"),
        ("category", "category (IPCC 2006)"),
        ("cat", "category (IPCC 2006)"),
        ("animal", "animal (FAOSTAT)"),
        ("product", "product (FAOSTAT)"),
        ("scenario", "scenario (FAOSTAT)"),
        ("scen", "scenario (FAOSTAT)"),
        ("provenance", "provenance"),
        ("model", "model"),
        ("source", "source"),
    ],
)
def test_pr_alias_array(opulent_ds, alias, full_name):
    da = opulent_ds.pr["CO2"]
    actual = da.pr.sum(dim=alias)
    expected = da.sum(dim=full_name, keep_attrs=True)
    xr.testing.assert_identical(actual, expected)


def test_pr_loc_select(opulent_ds):
    sel_pr = opulent_ds.pr.loc[
        {
            "time": slice("2002", "2005"),
            "area": ["COL", "ARG"],
            "animal": "cow",
        }
    ]
    sel = opulent_ds.loc[
        {
            "time": slice("2002", "2005"),
            "area (ISO3)": ["COL", "ARG"],
            "animal (FAOSTAT)": "cow",
        }
    ]
    xr.testing.assert_identical(sel_pr, sel)


def test_pr_loc_select_da(opulent_ds):
    da = opulent_ds["CO2"]
    sel_pr = da.pr.loc[
        {
            "time": slice("2002", "2005"),
            "area": ["COL", "ARG"],
            "animal": "cow",
        }
    ]
    sel = da.loc[
        {
            "time": slice("2002", "2005"),
            "area (ISO3)": ["COL", "ARG"],
            "animal (FAOSTAT)": "cow",
        }
    ]
    xr.testing.assert_identical(sel_pr, sel)
