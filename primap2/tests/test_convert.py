"""Tests for _convert.py"""

import climate_categories as cc
import xarray as xr

import primap2


def test_convert_ipcc(empty_ds: xr.Dataset):
    # build a DA categorized by IPCC1996 and with 1 everywhere so results are easy
    # to see
    da = empty_ds["CO2"]
    da = da.expand_dims({"category (IPCC1996)": list(cc.IPCC1996.keys())})
    arr = da.data.copy()
    arr[:] = 1 * primap2.ureg("Gg CO2 / year")
    da.data = arr

    da.pr.convert("category", "IPCC2006", sum_rule="extensive")

    # TODO test that values actually make sense
