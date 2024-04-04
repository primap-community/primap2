#!/usr/bin/env python
"""Tests for _units.py"""

import numpy as np
import pytest
import xarray as xr
import xarray.testing

from .utils import allclose, assert_equal


def test_roundtrip_quantify(opulent_ds: xr.Dataset):
    roundtrip = opulent_ds.pr.dequantify().pr.quantify()
    xarray.testing.assert_identical(roundtrip, opulent_ds)


def test_roundtrip_quantify_da(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6 (SARGWP100)"]
    roundtrip = da.pr.dequantify().pr.quantify()
    assert_equal(roundtrip, da)


def test_convert_to_gwp(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6"]
    da_converted = da.pr.convert_to_gwp("SARGWP100", "CO2 Gg / year")
    da_expected = opulent_ds["SF6 (SARGWP100)"]
    assert_equal(da_converted, da_expected)

    da_converted_like = da.pr.convert_to_gwp_like(da_expected)
    assert_equal(da_converted_like, da_expected)


def test_convert_to_gwp_like_missing(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6"]
    da_gwp = da.pr.convert_to_gwp("SARGWP100", "CO2 Gg / year")

    del da_gwp.attrs["gwp_context"]
    with pytest.raises(ValueError, match="reference array has no gwp_context"):
        da.pr.convert_to_gwp_like(da_gwp)

    da_gwp = xr.full_like(da_gwp, np.nan)
    da_gwp.attrs["gwp_context"] = "SARGWP100"
    with pytest.raises(ValueError, match="reference array has no units attached"):
        da.pr.convert_to_gwp_like(da_gwp)


def test_convert_to_gwp_incompatible(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6 (SARGWP100)"]
    with pytest.raises(ValueError, match="Incompatible gwp conversions"):
        da.pr.convert_to_gwp("AR5GWP", "CO2 Gg / year")


def test_convert_to_mass(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6 (SARGWP100)"]
    da_converted = da.pr.convert_to_mass()
    da_expected = opulent_ds["SF6"]
    assert_equal(da_converted, da_expected)


def test_convert_round_trip(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6"]
    assert da.attrs["entity"] == "SF6"
    da_gwp = da.pr.convert_to_gwp(gwp_context="AR4GWP100", units="Gg CO2 / year")
    da_rt = da_gwp.pr.convert_to_mass()
    assert_equal(da, da_rt)
    assert da_rt.attrs["entity"] == "SF6"
    assert isinstance(da_rt.attrs["entity"], str)


def test_convert_to_mass_missing_info(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6"]
    with pytest.raises(
        ValueError,
        match="No gwp_context given and no gwp_context available in the attrs",
    ):
        da.pr.convert_to_mass()

    da = opulent_ds["SF6 (SARGWP100)"]
    del da.attrs["entity"]
    with pytest.raises(
        ValueError, match="No entity given and no entity available in the attrs"
    ):
        da.pr.convert_to_mass()


def test_context(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6 (SARGWP100)"]
    with da.pr.gwp_context:
        da_converted = opulent_ds["SF6"].pint.to(da.pint.units)
    assert allclose(da, da_converted)
