"""Tests for _overview.py"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from primap2 import ureg


def test_to_df():
    data = np.array([[1, 2], [3, 4]], dtype=np.int64)
    a = ["a1", "a2"]
    b = ["b1", "b2"]
    da = xr.DataArray(data, coords=[("a", a), ("b", b)], name="name")
    actual = da.pr.to_df()

    expected = pd.DataFrame(data, index=a, columns=b)
    expected.index.name = "a"
    expected.columns.name = "b"

    pd.testing.assert_frame_equal(actual, expected)


def test_to_df_1d():
    data = np.array([1, 2], dtype=np.int64)
    a = ["a1", "a2"]
    da = xr.DataArray(data, coords=[("a", a)], name="name")
    actual = da.pr.to_df()

    expected = pd.Series(data, index=a, name="name")
    expected.index.name = "a"

    pd.testing.assert_series_equal(actual, expected)


def test_to_df_set():
    data = np.array([1, 2], dtype=np.int64)
    a = ["a1", "a2"]
    da = xr.DataArray(data, coords=[("a", a)], name="name")
    ds = xr.Dataset({"b": da})
    actual = ds.pr.to_df("name")

    expected = pd.DataFrame(data, index=a, columns=["b"])
    expected.index.name = "a"
    expected.columns.name = "name"

    pd.testing.assert_frame_equal(actual, expected)


def test_array_empty(empty_ds):
    with pytest.raises(ValueError, match="Specify at least one dimension"):
        empty_ds.pr.coverage()
    with pytest.raises(ValueError, match="Specify at least one dimension"):
        empty_ds["CO2"].pr.coverage()


def test_array_coverage(empty_ds):
    da = empty_ds["CO2"]
    da.pint.magnitude[:] = np.nan
    da.name = None

    da.pr.loc[{"time": "2001", "area": "COL"}] = 12.0 * ureg("Gg CO2 / year")
    da.pr.loc[{"time": "2002", "area": "COL"}] = 13.0 * ureg("Gg CO2 / year")

    expected = pd.DataFrame(
        index=da["area (ISO3)"].values,
        columns=da["time"].to_index(),
        data=np.zeros((len(da["area (ISO3)"]), len(da["time"])), dtype=np.int32),
    )
    expected.loc["COL", "2001"] = 1
    expected.loc["COL", "2002"] = 1
    expected.index.name = "area (ISO3)"
    expected.columns.name = "time"

    pd.testing.assert_frame_equal(
        expected.astype(np.int32), da.pr.coverage("area", "time").astype(np.int32)
    )
    pd.testing.assert_frame_equal(
        expected.T.astype(np.int32),
        da.pr.coverage("time", "area (ISO3)").astype(np.int32),
    )


def test_array_coverage_multidim(opulent_ds):
    da = opulent_ds["CO2"]

    da.pr.loc[{"product": "milk"}].pint.magnitude[:] = np.nan

    expected = pd.DataFrame(
        index=da.pr["animal"].values,
        columns=da.pr["product"].values,
        data=np.zeros((len(da.pr["animal"]), len(da.pr["product"])), dtype=np.int32),
    )
    expected[:] = np.prod(da.shape) // np.prod(expected.shape)
    expected.loc[:, "milk"] = 0
    expected.index.name = "animal (FAOSTAT)"
    expected.columns.name = "product (FAOSTAT)"

    pd.testing.assert_frame_equal(
        expected.astype(np.int32), da.pr.coverage("animal", "product").astype(np.int32)
    )
    pd.testing.assert_frame_equal(
        expected.T.astype(np.int32),
        da.pr.coverage("product", "animal").astype(np.int32),
    )


def test_array_coverage_error(opulent_ds):
    da = opulent_ds["CO2"]

    with pytest.raises(ValueError, match="Dimension 'non-existing' does not exist."):
        da.pr.coverage("animal", "non-existing")


def test_set_coverage(opulent_ds):
    ds = opulent_ds
    ds["CO2"].pr.loc[{"product": "milk"}].pint.magnitude[:] = np.nan

    expected = pd.DataFrame(
        index=ds.pr["product"].values,
        columns=ds.pr["animal"].values,
        data=np.zeros((len(ds.pr["product"]), len(ds.pr["animal"])), dtype=int),
    )
    expected[:] = np.prod(ds["CO2"].shape) // np.prod(expected.shape) * 4
    expected.loc["milk", :] = np.prod(ds["CO2"].shape) // np.prod(expected.shape) * 3
    expected.index.name = "product (FAOSTAT)"
    expected.columns.name = "animal (FAOSTAT)"
    expected.name = "coverage"

    pd.testing.assert_frame_equal(expected, ds.pr.coverage("product", "animal"))
    pd.testing.assert_frame_equal(expected.T, ds.pr.coverage("animal", "product"))


def test_set_coverage_entity(opulent_ds):
    ds = opulent_ds
    ds["CO2"].pr.loc[{"product": "milk"}].pint.magnitude[:] = np.nan

    expected = pd.DataFrame(
        index=list(ds.keys()),
        columns=ds.pr["area"].values,
        data=np.zeros((len(ds), len(ds.pr["area"].values)), dtype=int),
    )
    expected[:] = np.prod(ds["CO2"].shape)
    expected.loc["population", :] = np.prod(ds["population"].shape)
    expected.loc["CO2", :] = np.prod(ds["CO2"].shape) - np.prod(
        ds["CO2"].pr.loc[{"product": "milk"}].shape
    )
    expected = expected // len(ds.pr["area"].values)
    expected.name = "coverage"
    expected.index.name = "entity"
    expected.columns.name = "area (ISO3)"

    pd.testing.assert_frame_equal(expected, ds.pr.coverage("entity", "area"))


def test_set_coverage_boolean(opulent_ds):
    actual = opulent_ds.notnull().any("time").pr.coverage("entity", "area")
    expected = opulent_ds.pr.coverage("entity", "area") // len(opulent_ds["time"])

    pd.testing.assert_frame_equal(actual, expected)


def test_set_coverage_entity_other_dim_not_existing(opulent_ds):
    ds = opulent_ds

    ds["CO2"].pr.loc[{"product": "milk"}].pint.magnitude[:] = np.nan

    entites_expected = [x for x in ds.keys() if x != "population"]

    expected = pd.DataFrame(
        index=ds.pr["product"].values,
        columns=entites_expected,
        data=np.zeros((len(ds.pr["product"]), len(entites_expected)), dtype=int),
    )
    expected[:] = np.prod(ds["CO2"].shape) // len(ds.pr["product"])
    expected.loc["milk", "CO2"] = 0
    expected.index.name = "product (FAOSTAT)"
    expected.columns.name = "entity"

    pd.testing.assert_frame_equal(expected, ds.pr.coverage("product", "entity"))
    pd.testing.assert_frame_equal(expected.T, ds.pr.coverage("entity", "product"))


def test_set_coverage_error(opulent_ds):
    ds = opulent_ds["CO2"]

    with pytest.raises(ValueError, match="Dimension 'non-existing' does not exist."):
        ds.pr.coverage("animal", "non-existing")
