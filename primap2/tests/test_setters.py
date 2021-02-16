"""Tests for _setters.py"""

import re

import numpy as np
import pint
import pytest
import xarray as xr

from primap2 import ureg

from .utils import assert_elementwise_equal


@pytest.fixture
def da(minimal_ds) -> xr.DataArray:
    return minimal_ds["CO2"]


@pytest.fixture
def ts() -> np.ndarray:
    return np.linspace(0, 20, 21)


@pytest.fixture
def co2():
    return ureg("Gg CO2 / year")


class TestDASetter:
    def test_exists_default(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Values {'COL'} for 'area (ISO3)' already exist."
                " Use existing='overwrite' or 'fillna' to avoid this error."
            ),
        ):
            da.pr.set("area", "COL", ts * co2)

    def test_exists_overwrite(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        actual = da.pr.set("area", "COL", ts * co2, existing="overwrite")
        expected = da.copy()
        expected.loc[{"area (ISO3)": "COL"}] = ts[..., np.newaxis] * co2
        assert_elementwise_equal(actual, expected)

    def test_exists_fillna(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        expected = da.copy()
        expected.loc[{"area (ISO3)": "COL", "time": "2009"}] = np.nan * co2
        actual = expected.pr.set("area", "COL", ts * co2, existing="fillna")
        expected.loc[{"area (ISO3)": "COL", "time": "2009"}] = 9 * co2
        assert_elementwise_equal(actual, expected)

    def test_mixed_default(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Values {'COL'} for 'area (ISO3)' already exist."
                " Use existing='overwrite' or 'fillna' to avoid this error."
            ),
        ):
            da.pr.set("area", ["COL", "CUB"], ts * co2)

    def test_mixed_overwrite(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        actual = da.pr.set(
            "area",
            ["COL", "CUB"],
            np.array([ts, 2 * ts]).T * co2,
            existing="overwrite",
        )
        expected = da.reindex({"area (ISO3)": list(da["area (ISO3)"].values) + ["CUB"]})
        expected.loc[{"area (ISO3)": "COL"}] = ts[..., np.newaxis] * co2
        expected.loc[{"area (ISO3)": "CUB"}] = ts[..., np.newaxis] * 2 * co2
        assert_elementwise_equal(actual, expected)

        # with explicit value_dims
        actual = da.pr.set(
            "area",
            ["COL", "CUB"],
            np.array([ts, 2 * ts]) * co2,
            value_dims=["area (ISO3)", "time"],
            existing="overwrite",
        )
        assert_elementwise_equal(actual, expected)

    def test_mixed_fillna(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        tda = da.copy()
        tda.loc[{"area (ISO3)": "COL", "time": "2009"}] = np.nan * co2
        actual = tda.pr.set(
            "area",
            ["COL", "CUB"],
            np.array([ts, 2 * ts]).T * co2,
            existing="fillna",
        )
        expected = da.reindex({"area (ISO3)": list(da["area (ISO3)"].values) + ["CUB"]})
        expected.loc[{"area (ISO3)": "COL", "time": "2009"}] = 9 * co2
        expected.loc[{"area (ISO3)": "CUB"}] = ts[..., np.newaxis] * 2 * co2
        assert_elementwise_equal(actual, expected)

    def test_new_from_array(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        # with scalar dimension
        actual = da.pr.set("area", "CUB", 2 * da.pr.loc[{"area": "COL"}])
        expected = da.reindex({"area (ISO3)": list(da["area (ISO3)"].values) + ["CUB"]})
        expected = expected.fillna(expected.loc[{"area (ISO3)": "COL"}] * 2)
        assert_elementwise_equal(actual, expected)

        # with non-scalar dimension
        with pytest.raises(KeyError, match="not all values found in index"):
            da.pr.set("area", "CUB", 2 * da.pr.loc[{"area": ["COL"]}])

    def test_mixed_from_data_array_overwrite(
        self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit
    ):
        actual = da.pr.set(
            "area", ["CUB", "COL"], 2 * da.pr.loc[{"area": "COL"}], existing="overwrite"
        )
        expected = da.reindex({"area (ISO3)": list(da["area (ISO3)"].values) + ["CUB"]})
        expected.loc[{"area (ISO3)": "COL"}] = np.nan
        expected = expected.fillna(da.loc[{"area (ISO3)": "COL"}] * 2)
        assert_elementwise_equal(actual, expected)

    @pytest.fixture
    def mda(self):
        return xr.DataArray(
            np.zeros((3, 4, 5)),
            coords=[
                ("a", ["a1", "a2", "a3"]),
                ("b", ["b1", "b2", "b3", "b4"]),
                ("c", ["c1", "c2", "c3", "c4", "c5"]),
            ],
            attrs={"test": "attr"},
            name="thing",
        )

    def test_multidimensional_ndarray(self, mda: xr.DataArray):
        actual = mda.pr.set("a", "a4", np.ones((4, 5)))
        expected = mda.reindex({"a": ["a1", "a2", "a3", "a4"]}).fillna(1)
        assert_elementwise_equal(actual, expected)

        actual = mda.pr.set("a", "a4", np.ones(4), value_dims=["b"])
        assert_elementwise_equal(actual, expected)

        actual = mda.pr.set("a", "a4", np.ones(5), value_dims=["c"])
        assert_elementwise_equal(actual, expected)

        actual = mda.pr.set("a", "a4", np.ones((4, 5)), value_dims=["b", "c"])
        assert_elementwise_equal(actual, expected)

    def test_multidimensional_ndarray_underspecified(self, mda: xr.DataArray):
        match = (
            "Could not automatically determine value dimensions, please use the"
            " value_dims parameter."
        )
        with pytest.raises(ValueError, match=match):
            mda.pr.set("a", "a4", np.ones(5))
        with pytest.raises(ValueError, match=match):
            mda.pr.set("a", "a4", np.ones(4))

    def test_multidimensional_data_array(self, mda: xr.DataArray):
        actual = mda.pr.set(
            "a", "a4", xr.DataArray(np.ones((4, 5)), coords=[mda["b"], mda["c"]])
        )
        expected = mda.reindex({"a": ["a1", "a2", "a3", "a4"]}).fillna(1)
        assert_elementwise_equal(actual, expected)

        actual = mda.pr.set("a", "a4", xr.DataArray(np.ones(4), coords=[mda["b"]]))
        assert_elementwise_equal(actual, expected)

        actual = mda.pr.set("a", "a4", xr.DataArray(np.ones(5), coords=[mda["c"]]))
        assert_elementwise_equal(actual, expected)

    def test_multidimensional_data_array_dim_contained(self, mda: xr.DataArray):
        actual = mda.pr.set(
            "a",
            ["a3", "a4", "a5"],
            xr.DataArray(
                np.ones((3, 4, 5)),
                coords=[("a", ["a3", "a4", "a5"]), mda["b"], mda["c"]],
            ),
            existing="overwrite",
        )
        expected = mda.reindex({"a": ["a1", "a2", "a3", "a4", "a5"]}).fillna(1)
        expected.loc[{"a": "a3"}] = 1
        assert_elementwise_equal(actual, expected)

    def test_multidimensional_data_array_dim_oversized(self, mda: xr.DataArray):
        actual = mda.pr.set(
            "a",
            ["a3", "a4", "a5"],
            xr.DataArray(
                np.ones((4, 4, 5)),
                coords=[("a", ["a2", "a3", "a4", "a5"]), mda["b"], mda["c"]],
            ),
            existing="overwrite",
        )
        expected = mda.reindex({"a": ["a1", "a2", "a3", "a4", "a5"]}).fillna(1)
        expected.loc[{"a": "a3"}] = 1
        assert_elementwise_equal(actual, expected)


@pytest.mark.parametrize(
    ["dim", "existing", "error", "match"],
    [
        ("asdf", "error", ValueError, "Dimension 'asdf' does not exist."),
        (
            "area",
            "asdf",
            ValueError,
            "If given, 'existing' must specify one of 'error', 'overwrite', or"
            " 'fillna', not 'asdf'.",
        ),
    ],
)
def test_da_setter_errors(da: xr.DataArray, dim, existing, error, match):
    with pytest.raises(error, match=match):
        da.pr.set(dim, ["COL"], np.linspace(0, 20, 21), existing=existing)


def test_da_setter_overspecifiec(da: xr.DataArray, ts: np.ndarray):
    with pytest.raises(
        ValueError, match="value_dims given, but value is already a DataArray."
    ):
        da.pr.set(
            "area",
            "COL",
            da.pr.loc[{"area": "BOL"}],
            value_dims=["area", "time", "source"],
            existing="overwrite",
        )