"""Simple selection and loc-style accessor which automatically translates PRIMAP2 short
column names to the actual long names including the categorization."""

import typing

import xarray as xr

from . import _accessor_base


@typing.overload
def translate(item: str, translations: typing.Dict[str, str]) -> str:
    ...


@typing.overload
def translate(
    item: typing.Mapping[str, typing.Any], translations: typing.Dict[str, str]
) -> typing.Mapping[str, typing.Any]:
    ...


def translate(item, translations):
    if isinstance(item, str):
        if item in translations:
            return translations[item]
        else:
            return item
    else:
        sel = {}
        for key in item:
            if key in translations:
                sel[translations[key]] = item[key]
            else:
                sel[key] = item[key]
        return sel


class AliasLocIndexer:
    """Provides loc-style selection with aliases. Needs to be a separate class for
    __getitem__ functionality, which doesn't work directly on properties without an
    intermediate object."""

    __slots__ = ("_obj",)

    def __init__(self, obj: typing.Union[xr.Dataset, xr.DataArray]):
        self._obj = obj

    def __getitem__(self, item: typing.Mapping[str, typing.Any]) -> xr.Dataset:
        return self._obj.loc[translate(item, self._obj.pr._get_translations())]


class DataArrayAliasSelectionAccessor(_accessor_base.BaseDataArrayAccessor):
    def _get_translations(self) -> typing.Dict[str, str]:
        # we have to do string parsing because the Dataset's attrs are not available
        # in the DataArray context
        ret = {}
        for dim in self._da.dims:
            if " (" in dim:
                key: str = dim.split("(")[0][:-1]
                ret[key] = dim
        return ret

    @property
    def loc(self):
        """Attribute for location-based indexing like xr.DataArray.loc, but also
        supports short aliases like ``area`` and translates them into the long
        names including the corresponding category-set."""
        return AliasLocIndexer(self._da)


class DatasetAliasSelectionAccessor(_accessor_base.BaseDatasetAccessor):
    def _get_translations(self) -> typing.Dict[str, str]:
        ret = {"area": self._ds.attrs["area"]}  # required key
        for key, abbrev in [("category", "cat"), ("scenario", "scen")]:
            if abbrev in self._ds.attrs:
                ret[key] = self._ds.attrs[abbrev]
        if "sec_cats" in self._ds.attrs:
            for full_name in self._ds.attrs["sec_cats"]:
                key = full_name.split("(")[0][:-1]
                ret[key] = full_name
        return ret

    @typing.overload
    def __getitem__(self, item: str) -> xr.DataArray:
        ...

    @typing.overload
    def __getitem__(self, item: typing.Mapping[str, typing.Any]) -> xr.Dataset:
        ...

    def __getitem__(self, item):
        """Like ds[], but translates short aliases like "area" into the long names
        including the corresponding category-set."""
        return self._ds[translate(item, self._get_translations())]

    @property
    def loc(self):
        """Attribute for location-based indexing like xr.Dataset.loc, but also
        supports short aliases like ``area`` and translates them into the long
        names including the corresponding category-set."""
        return AliasLocIndexer(self._ds)
