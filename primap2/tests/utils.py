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
