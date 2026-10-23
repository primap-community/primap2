import numpy as np
import pandas as pd
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
    dimensions aligned and sorted equally so that naive comparisons can be done.
    """
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
    for key in a:
        assert_aligned_equal(a[key], b[key], *args, **kwargs)
    assert a.attrs == b.attrs, (a.attrs, b.attrs)


def description_fields(description) -> tuple:
    """The fields of a ProcessingStepDescription, which is compared by identity."""
    return (
        description.function,
        description.description,
        description.source,
        description.variable,
        None if description.coords is None else tuple(description.coords.items()),
    )


def step_fields(step, _memo: dict | None = None) -> tuple:
    """The description fields, time and (recursively) parents of a ProcessingStep."""
    memo = {} if _memo is None else _memo
    if id(step) not in memo:
        memo[id(step)] = (
            description_fields(step.description),
            step.time,
            tuple(step_fields(p, memo) for p in step.parents),
        )
    return memo[id(step)]


def assert_steps_equal(a, b):
    """Assert that two processing steps have equal fields, time and parents."""
    assert step_fields(a) == step_fields(b)


def assert_ds_identical(a: xr.Dataset, b: xr.Dataset):
    """Like xr.testing.assert_identical, comparing processing steps by their fields."""
    names = [n for n in a.data_vars if str(n).startswith("Processing of ")]
    assert names == [n for n in b.data_vars if str(n).startswith("Processing of ")]
    xr.testing.assert_identical(a.drop_vars(names), b.drop_vars(names))
    for name in names:
        assert a[name].dims == b[name].dims
        assert a[name].attrs == b[name].attrs
        for x, y in zip(
            a[name].values.ravel().tolist(), b[name].values.ravel().tolist(), strict=True
        ):
            if pd.isnull(x) or pd.isnull(y):
                assert pd.isnull(x) and pd.isnull(y)
            else:
                assert_steps_equal(x, y)
