from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr


def get_single_ts(
    *,
    time: pd.DatetimeIndex | None = None,
    data: np.ndarray | None = None,
    dims: Sequence[str] | None = None,
    coords: dict[str, str | Sequence[str]] | None = None,
) -> xr.DataArray:
    if time is None:
        time = pd.date_range("1850-01-01", "2022-01-01", freq="YS")
    if dims is None:
        dims = []
    if data is None:
        data = np.linspace(0.0, 1.0, len(time))
    if coords is None:
        coords = {}
    return xr.DataArray(
        data,
        dims=["time", *dims],
        coords={"time": time, **coords},
    )
