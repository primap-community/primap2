"""Simple selection and loc-style accessor which automatically translates PRIMAP2 short
column names to the actual long names including the categorization."""

import typing

import xarray as xr

from . import _accessor_base

KeyT = typing.TypeVar("KeyT", str, typing.Mapping[typing.Hashable, typing.Any])


def translate(item: KeyT, translations: typing.Mapping[typing.Hashable, str]) -> KeyT:
    if isinstance(item, str):
        if item in translations:
            return translations[item]
        else:
            return item
    else:
        sel: typing.Dict[typing.Hashable, typing.Hashable] = {}
        for key in item:
            if key in translations:
                sel[translations[key]] = item[key]
            else:
                sel[key] = item[key]
        return sel


class DataArrayAliasLocIndexer:
    """Provides loc-style selection with aliases. Needs to be a separate class for
    __getitem__ and __setitem__ functionality, which doesn't work directly on properties
    without an intermediate object."""

    __slots__ = ("_da",)

    def __init__(self, da: xr.DataArray):
        self._da = da

    def __getitem__(
        self, item: typing.Mapping[typing.Hashable, typing.Any]
    ) -> xr.DataArray:
        return self._da.loc[translate(item, self._da.pr.dim_alias_translations)]

    def __setitem__(self, key: typing.Mapping[typing.Hashable, typing.Any], value):
        self._da.loc.__setitem__(
            translate(key, self._da.pr.dim_alias_translations), value
        )


class DataArrayAliasSelectionAccessor(_accessor_base.BaseDataArrayAccessor):
    @property
    def dim_alias_translations(self) -> typing.Dict[typing.Hashable, str]:
        """Translate a shortened dimension alias to a full dimension name.

        For example, if the full dimension name is ``area (ISO3)``, the alias ``area``
        is mapped to ``area (ISO3)``.

        Returns
        -------
        translations : dict
            A mapping of all dimension aliases to full dimension names.
        """
        # we have to do string parsing because the Dataset's attrs are not available
        # in the DataArray context
        ret: typing.Dict[typing.Hashable, str] = {}
        for dim in self._da.dims:
            if isinstance(dim, str):
                if " (" in dim:
                    key: str = dim.split("(")[0][:-1]
                    ret[key] = dim
        return ret

    @property
    def loc(self):
        """Attribute for location-based indexing like xr.DataArray.loc, but also
        supports short aliases like ``area`` and translates them into the long
        names including the corresponding category-set."""
        return DataArrayAliasLocIndexer(self._da)


class DatasetAliasLocIndexer:
    """Provides loc-style selection with aliases. Needs to be a separate class for
    __getitem__ functionality, which doesn't work directly on properties without an
    intermediate object."""

    __slots__ = ("_ds",)

    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def __getitem__(
        self, item: typing.Mapping[typing.Hashable, typing.Any]
    ) -> xr.Dataset:
        return self._ds.loc[translate(item, self._ds.pr.dim_alias_translations)]


class DatasetAliasSelectionAccessor(_accessor_base.BaseDatasetAccessor):
    @property
    def dim_alias_translations(self) -> typing.Dict[typing.Hashable, str]:
        """Translate a shortened dimension alias to a full dimension name.

        For example, if the full dimension name is ``area (ISO3)``, the alias ``area``
        is mapped to ``area (ISO3)``.

        Returns
        -------
        translations : dict
            A mapping of all dimension aliases to full dimension names.
        """
        ret: typing.Dict[typing.Hashable, str] = {}
        for key, abbrev in [
            ("category", "cat"),
            ("scenario", "scen"),
            ("area", "area"),
        ]:
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
        return self._ds[translate(item, self.dim_alias_translations)]

    @property
    def loc(self):
        """Attribute for location-based indexing like xr.Dataset.loc, but also
        supports short aliases like ``area`` and translates them into the long
        names including the corresponding category-set."""
        return DatasetAliasLocIndexer(self._ds)
