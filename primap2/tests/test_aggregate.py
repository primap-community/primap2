#!/usr/bin/env python
"""Tests for _aggregate.py"""

import numpy as np
import xarray as xr
import xarray.testing


def test_fill_all_na():
    coords = [("a", [1, 2]), ("b", [1, 2]), ("c", [1, 2, 3])]
    da = xr.DataArray(
        data=[
            [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
            [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
        ],
        coords=coords,
    )
    a = da.pr.fill_all_na(dim="b", value=0)
    a_expected = xr.DataArray(
        data=[
            [[0, np.nan, np.nan], [0, 1, 2]],
            [[0, np.nan, np.nan], [0, 1, 2]],
        ],
        coords=coords,
    )
    assert np.allclose(a, a_expected, equal_nan=True)

    b = da.pr.fill_all_na(dim=("a", "b"), value=0)
    assert np.allclose(b, a_expected, equal_nan=True)

    c = da.pr.fill_all_na(dim=("b", "c"), value=0)
    assert np.allclose(c, da, equal_nan=True)

    d = da.pr.fill_all_na(dim="c", value=0)
    d_expected = xr.DataArray(
        data=[
            [[0, 0, 0], [np.nan, 1, 2]],
            [[0, 0, 0], [np.nan, 1, 2]],
        ],
        coords=coords,
    )
    assert np.allclose(d, d_expected, equal_nan=True)

    ds = xr.Dataset({"1": da, "2": da.copy()})
    dsf = ds.pr.fill_all_na(dim="b", value=0)
    assert np.allclose(dsf["1"], a_expected, equal_nan=True)
    assert np.allclose(dsf["2"], a_expected, equal_nan=True)


def test_sum_skip_allna():
    coords = [("a", [1, 2]), ("b", [1, 2]), ("c", [1, 2, 3])]
    da = xr.DataArray(
        data=[
            [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
            [[np.nan, np.nan, np.nan], [np.nan, 1, 2]],
        ],
        coords=coords,
    )

    a = da.pr.sum_skip_allna(dim="a")
    a_expected = xr.DataArray(
        data=[
            [np.nan, np.nan, np.nan],
            [np.nan, 2, 4],
        ],
        coords=coords[1:],
    )
    assert np.allclose(a, a_expected, equal_nan=True)

    b = da.pr.sum_skip_allna(dim="b", skipna_evaluation_dims="a")
    b_expected = xr.DataArray(
        data=[[0, 1, 2], [0, 1, 2]], coords=[coords[0], coords[2]]
    )
    assert np.allclose(b, b_expected, equal_nan=True)

    c = da.pr.sum_skip_allna(dim="b", skipna_evaluation_dims="c")
    c_expected = xr.DataArray(
        data=[[np.nan, 1, 2], [np.nan, 1, 2]], coords=[coords[0], coords[2]]
    )
    assert np.allclose(c, c_expected, equal_nan=True)

    ds = xr.Dataset({"1": da, "2": da.copy()})
    dss = ds.pr.sum_skip_allna(dim="a")
    dss_expected = xr.Dataset(
        {
            "1": xr.DataArray(
                data=[
                    [np.nan, np.nan, np.nan],
                    [np.nan, 2, 4],
                ],
                coords=coords[1:],
            ),
            "2": xr.DataArray(
                data=[
                    [np.nan, np.nan, np.nan],
                    [np.nan, 2, 4],
                ],
                coords=coords[1:],
            ),
        }
    )
    xr.testing.assert_identical(dss, dss_expected)


# TODO: missing: gas_basket_contents_sum, fill_na_gas_basket_from_contents
