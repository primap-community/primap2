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
    assert a.attrs == b.attrs
    assert a.name == b.name


def assert_align(a: xr.DataArray, b: xr.DataArray) -> (xr.DataArray, xr.DataArray):
    aa, ba = xr.align(a, b, join="outer")
    size_unchanged = sorted(aa.shape) == sorted(a.shape) and sorted(ba.shape) == sorted(
        b.shape
    )
    assert size_unchanged, (a.shape, b.shape)
    return aa, ba


def assert_elementwise_equal(a: xr.DataArray, b: xr.DataArray):
    a, b = assert_align(a, b)
    assert np.all(a == b)
    assert a.attrs == b.attrs
    assert a.name == b.name


def assert_ds_elementwise_equal(a: xr.Dataset, b: xr.Dataset):
    assert set(a.keys()) == set(b.keys())
    for key in a.keys():
        assert_elementwise_equal(a[key], b[key])
    assert a.attrs == b.attrs
