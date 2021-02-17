import numpy as np
import pint
import xarray as xr


def allclose(a: xr.DataArray, b: xr.DataArray, *args, **kwargs):
    try:
        a = a.pint.to(b.pint.units)
    except pint.DimensionalityError:
        return False
    return np.allclose(a.pint.magnitude, b.pint.magnitude, *args, **kwargs)


def assert_equal(a: xr.DataArray, b: xr.DataArray, *args, **kwargs):
    assert allclose(a, b, *args, **kwargs)
    assert a.attrs == b.attrs, (a.attrs, b.attrs)
    assert a.name == b.name, (a.name, b.name)


def assert_align(a: xr.DataArray, b: xr.DataArray) -> (xr.DataArray, xr.DataArray):
    aa, ba = xr.align(a, b, join="outer")
    aa = aa.transpose(*ba.dims)
    size_unchanged = sorted(aa.shape) == sorted(a.shape) and ba.shape == b.shape
    assert size_unchanged, (a.shape, b.shape)
    return aa, ba


def assert_aligned_equal(a: xr.DataArray, b: xr.DataArray, *args, **kwargs):
    a, b = assert_align(a, b)
    assert_equal(a, b, *args, **kwargs)


def assert_ds_aligned_equal(a: xr.Dataset, b: xr.Dataset, *args, **kwargs):
    assert set(a.keys()) == set(b.keys())
    for key in a.keys():
        assert_aligned_equal(a[key], b[key], *args, **kwargs)
    assert a.attrs == b.attrs, (a.attrs, b.attrs)
