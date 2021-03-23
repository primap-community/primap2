"""Tests for _overview.py"""


import numpy as np
import pandas as pd
import pytest

from primap2 import ureg


def test_array_coverage(empty_ds):
    da = empty_ds["CO2"]
    da[:] = np.nan

    da.pr.loc[{"time": "2001", "area": "COL"}] = 12.0 * ureg("Gg CO2 / year")
    da.pr.loc[{"time": "2002", "area": "COL"}] = 13.0 * ureg("Gg CO2 / year")

    expected = pd.DataFrame(
        index=da["time"].values,
        columns=da["area (ISO3)"].values,
        data=np.zeros((len(da["time"]), len(da["area (ISO3)"])), dtype=int),
    )
    expected.loc["2001", "COL"] = 1
    expected.loc["2002", "COL"] = 1
    expected.index.name = "time"
    expected.columns.name = "area (ISO3)"

    pd.testing.assert_frame_equal(expected, da.pr.coverage("area", "time"))
    pd.testing.assert_frame_equal(expected.T, da.pr.coverage("time", "area (ISO3)"))


def test_array_coverage_multidim(opulent_ds):
    da = opulent_ds["CO2"]

    da.pr.loc[{"product": "milk"}] = np.nan

    expected = pd.DataFrame(
        index=da.pr["product"].values,
        columns=da.pr["animal"].values,
        data=np.zeros((len(da.pr["product"]), len(da.pr["animal"])), dtype=int),
    )
    expected[:] = np.product(da.shape) // np.product(expected.shape)
    expected.loc["milk", :] = 0
    expected.index.name = "product (FAOSTAT)"
    expected.columns.name = "animal (FAOSTAT)"

    pd.testing.assert_frame_equal(expected, da.pr.coverage("animal", "product"))
    pd.testing.assert_frame_equal(expected.T, da.pr.coverage("product", "animal"))


def test_array_coverage_error(opulent_ds):
    da = opulent_ds["CO2"]

    with pytest.raises(ValueError, match="'non-existing' is not a dimension."):
        da.pr.coverage("animal", "non-existing")


def test_set_coverage(opulent_ds):
    ds = opulent_ds

    ds["CO2"].pr.loc[{"product": "milk"}] = np.nan

    expected = pd.DataFrame(
        index=ds.pr["product"].values,
        columns=ds.pr["animal"].values,
        data=np.zeros((len(ds.pr["product"]), len(ds.pr["animal"])), dtype=int),
    )
    expected[:] = np.product(ds["CO2"].shape) // np.product(expected.shape) * 4
    expected.loc["milk", :] = (
        np.product(ds["CO2"].shape) // np.product(expected.shape) * 3
    )
    expected.index.name = "product (FAOSTAT)"
    expected.columns.name = "animal (FAOSTAT)"

    pd.testing.assert_frame_equal(expected, ds.pr.coverage("animal", "product"))
    pd.testing.assert_frame_equal(expected.T, ds.pr.coverage("product", "animal"))


def test_set_coverage_entity(opulent_ds):
    ds = opulent_ds

    ds["CO2"].pr.loc[{"product": "milk"}] = np.nan

    entites_expected = [x for x in ds.keys() if x != "population"]

    expected = pd.DataFrame(
        index=ds.pr["product"].values,
        columns=entites_expected,
        data=np.zeros((len(ds.pr["product"]), len(entites_expected)), dtype=int),
    )
    expected[:] = np.product(ds["CO2"].shape) // len(ds.pr["product"])
    expected.loc["milk", "CO2"] = 0
    expected.index.name = "product (FAOSTAT)"
    expected.columns.name = "entity"

    pd.testing.assert_frame_equal(expected, ds.pr.coverage("entity", "product"))
    pd.testing.assert_frame_equal(expected.T, ds.pr.coverage("product", "entity"))


def test_set_coverage_error(opulent_ds):
    ds = opulent_ds["CO2"]

    with pytest.raises(ValueError, match="'non-existing' is not a dimension"):
        ds.pr.coverage("animal", "non-existing")
