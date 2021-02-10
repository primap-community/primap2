"""Tests for _loc_accessor.py"""

import pytest
import xarray as xr
import xarray.testing


@pytest.mark.parametrize(
    ["alias", "full_name"],
    [
        ("time", "time"),
        ("area", "area (ISO3)"),
        ("category", "category (FAOSTAT)"),
        ("animal", "animal (FAOSTAT)"),
        ("product", "product (FAOSTAT)"),
        ("scenario", "scenario (FAOSTAT)"),
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
