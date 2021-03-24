import typing

import xarray as xr

DatasetOrDataArray = typing.TypeVar("DatasetOrDataArray", xr.Dataset, xr.DataArray)
KeyT = typing.TypeVar("KeyT", str, typing.Mapping[typing.Hashable, typing.Any])
DimOrDimsT = typing.TypeVar(
    "DimOrDimsT",
    str,
    typing.Hashable,
    typing.Iterable[str],
    typing.Iterable[typing.Hashable],
)
FunctionT = typing.TypeVar("FunctionT", bound=typing.Callable[..., typing.Any])
