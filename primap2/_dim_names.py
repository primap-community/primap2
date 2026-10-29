import typing

import pandas as pd
import xarray as xr

from primap2._types import DatasetOrDataArray


def dim_names(obj: DatasetOrDataArray) -> tuple[typing.Hashable]:
    """Extract the names of dimensions compatible with all xarray versions."""
    return obj.sizes.keys()


def format_scalar_dims(da: xr.DataArray) -> str:
    """Describe the length-1 dimensions of ``da`` like ``"area=COL, time=2000"``."""
    scalar_dims_format = []
    for dim in da.dims:
        if len(da[dim]) != 1:
            continue
        if pd.api.types.is_datetime64_any_dtype(da[dim]):
            ts = pd.Timestamp(da[dim][0].values)
            # optimization for the common case where we have data on a yearly or
            # more coarse basis
            if ts == pd.Timestamp(year=ts.year, month=1, day=1):
                scalar_dims_format.append(f"{dim}={ts.strftime('%Y')}")
            else:
                scalar_dims_format.append(f"{dim}={ts!s}")
        else:
            scalar_dims_format.append(f"{dim}={da[dim].item()}")
    return ", ".join(scalar_dims_format)
