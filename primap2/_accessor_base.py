"""Base classes for accessor mix-in classes."""

import typing

import xarray as xr

XrObj = typing.TypeVar("XrObj", xr.Dataset, xr.Dataset)


class BaseDataArrayAccessor:
    def __init__(self, da: xr.DataArray):
        self._da = da


class BaseDatasetAccessor:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds
