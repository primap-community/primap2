"""Tests for _alias_selection.py"""

import pytest
import xarray as xr
import xarray.testing

import primap2


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


def test_pr_loc_select_not(opulent_ds):
    sel_pr = opulent_ds.pr.loc[
        {
            "time": slice("2002", "2005"),
            "area": ["COL", "ARG"],
            "animal": primap2.Not("cow"),
            "category": primap2.Not(["0", "1"]),
        }
    ]
    sel = opulent_ds.loc[
        {
            "time": slice("2002", "2005"),
            "area (ISO3)": ["COL", "ARG"],
            "animal (FAOSTAT)": ["swine", "goat"],
            "category (IPCC 2006)": ["2", "3", "4", "5", "1.A", "1.B"],
        }
    ]
    xr.testing.assert_identical(sel_pr, sel)


def test_pr_loc_select_da_not(opulent_ds):
    da = opulent_ds["CO2"]
    sel_pr = da.pr.loc[
        {
            "time": slice("2002", "2005"),
            "area": ["COL", "ARG"],
            "animal": primap2.Not("cow"),
            "category": primap2.Not(["0", "1"]),
        }
    ]
    sel = da.loc[
        {
            "time": slice("2002", "2005"),
            "area (ISO3)": ["COL", "ARG"],
            "animal (FAOSTAT)": ["swine", "goat"],
            "category (IPCC 2006)": ["2", "3", "4", "5", "1.A", "1.B"],
        }
    ]
    xr.testing.assert_identical(sel_pr, sel)


def test_resolve_not(opulent_ds):
    result = primap2._selection.resolve_not(
        input_selector={
            "a": "1",
            "b": ["1", "2"],
            "animal (FAOSTAT)": primap2.Not("cow"),
            "area (ISO3)": primap2.Not(["MEX", "COL"]),
        },
        xarray_obj=opulent_ds,
    )
    assert len(result) == 4
    assert result["a"] == "1"
    assert result["b"] == ["1", "2"]
    assert len(result["animal (FAOSTAT)"]) == 2
    assert "swine" in result["animal (FAOSTAT)"]
    assert "goat" in result["animal (FAOSTAT)"]
    assert len(result["area (ISO3)"]) == 2
    assert "ARG" in result["area (ISO3)"]
    assert "BOL" in result["area (ISO3)"]


def test_resolve_not_da(opulent_ds):
    result = primap2._selection.resolve_not(
        input_selector={
            "a": "1",
            "b": ["1", "2"],
            "animal (FAOSTAT)": primap2.Not("cow"),
            "area (ISO3)": primap2.Not(["MEX", "COL"]),
        },
        xarray_obj=opulent_ds["CO2"],
    )
    assert len(result) == 4
    assert result["a"] == "1"
    assert result["b"] == ["1", "2"]
    assert len(result["animal (FAOSTAT)"]) == 2
    assert "swine" in result["animal (FAOSTAT)"]
    assert "goat" in result["animal (FAOSTAT)"]
    assert len(result["area (ISO3)"]) == 2
    assert "ARG" in result["area (ISO3)"]
    assert "BOL" in result["area (ISO3)"]
