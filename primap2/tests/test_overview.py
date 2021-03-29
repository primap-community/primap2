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
        index=da["area (ISO3)"].values,
        columns=da["time"].values,
        data=np.zeros((len(da["area (ISO3)"]), len(da["time"])), dtype=int),
    )
    expected.loc["COL", "2001"] = 1
    expected.loc["COL", "2002"] = 1
    expected.index.name = "area (ISO3)"
    expected.columns.name = "time"

    pd.testing.assert_frame_equal(expected, da.pr.coverage("area", "time"))
    pd.testing.assert_frame_equal(expected.T, da.pr.coverage("time", "area (ISO3)"))


def test_array_coverage_multidim(opulent_ds):
    da = opulent_ds["CO2"]

    da.pr.loc[{"product": "milk"}] = np.nan

    expected = pd.DataFrame(
        index=da.pr["animal"].values,
        columns=da.pr["product"].values,
        data=np.zeros((len(da.pr["animal"]), len(da.pr["product"])), dtype=int),
    )
    expected[:] = np.product(da.shape) // np.product(expected.shape)
    expected.loc[:, "milk"] = 0
    expected.index.name = "animal (FAOSTAT)"
    expected.columns.name = "product (FAOSTAT)"

    pd.testing.assert_frame_equal(expected, da.pr.coverage("animal", "product"))
    pd.testing.assert_frame_equal(expected.T, da.pr.coverage("product", "animal"))


def test_array_coverage_error(opulent_ds):
    da = opulent_ds["CO2"]

    with pytest.raises(ValueError, match="Dimension 'non-existing' does not exist."):
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

    pd.testing.assert_frame_equal(expected, ds.pr.coverage("product", "animal"))
    pd.testing.assert_frame_equal(expected.T, ds.pr.coverage("animal", "product"))


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

    pd.testing.assert_frame_equal(expected, ds.pr.coverage("product", "entity"))
    pd.testing.assert_frame_equal(expected.T, ds.pr.coverage("entity", "product"))


def test_set_coverage_error(opulent_ds):
    ds = opulent_ds["CO2"]

    with pytest.raises(ValueError, match="Dimension 'non-existing' does not exist."):
        ds.pr.coverage("animal", "non-existing")
