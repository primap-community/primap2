import numpy as np
import pint
import xarray as xr


def allclose(a: xr.DataArray, b: xr.DataArray, *args, **kwargs) -> bool:
    """Like np.allclose, but converts a to b's units before comparing."""
    try:
        a = a.pint.to(b.pint.units)
    except pint.DimensionalityError:
        return False
    if a.dtype == float:  # need to use "allclose" to compare floats
        return np.allclose(a.pint.magnitude, b.pint.magnitude, *args, **kwargs)
    else:
        return (a.pint.magnitude == b.pint.magnitude).all()


def assert_equal(a: xr.DataArray, b: xr.DataArray, *args, **kwargs):
    """Asserts that contents are allclose(), and the name and attrs are also equal."""
    assert allclose(a, b, *args, **kwargs)
    assert a.attrs == b.attrs, (a.attrs, b.attrs)
    assert a.name == b.name, (a.name, b.name)


def assert_align(a: xr.DataArray, b: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Asserts that a and b have the same shape and returns a and b with axes and
    dimensions aligned and sorted equally so that naive comparisons can be done."""
    assert set(a.dims) == set(b.dims), (a.dims, b.dims)
    aa, ba = xr.align(a, b, join="outer")
    aa = aa.transpose(*ba.dims)
    size_unchanged = sorted(aa.shape) == sorted(a.shape) and ba.shape == b.shape
    assert size_unchanged, (a.shape, b.shape)
    return aa, ba


def assert_aligned_equal(a: xr.DataArray, b: xr.DataArray, *args, **kwargs):
    """Assert that a and b are equal after alignment of their dimensions."""
    a, b = assert_align(a, b)
    assert_equal(a, b, *args, **kwargs)


def assert_ds_aligned_equal(a: xr.Dataset, b: xr.Dataset, *args, **kwargs):
    """Assert that a and b are equal after alignment of their dimensions."""
    assert set(a.keys()) == set(b.keys())
    for key in a.keys():
        assert_aligned_equal(a[key], b[key], *args, **kwargs)
    assert a.attrs == b.attrs, (a.attrs, b.attrs)
