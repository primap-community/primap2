import numpy as np
import xarray as xr


def variance_detrend(
    ts: xr.DataArray,
    average_period: int = 11,
) -> xr.DataArray:
    """
    Detrend time-series and calculate the variance

    This function first detrends the time-series by subtracting a rolling
    mean using a centered window of length `average_period`. Then the
    variance is calculated and returned. NaN values are ignored in the
    process.

    The ts DataArray can be multi-dimensional. The variance is calculated
    over the time dimension.

    Parameters
    ----------
    ts
        time series to calculate the rolling mean from
    average_period
        length of the averaging period

    Returns
    -------
        xr.DataArray with the variance

    """

    nan_mask = np.isnan(ts)
    ts_mean = ts.rolling(time=average_period, min_periods=1, center=True).mean()
    ts_mean[nan_mask] = np.nan
    ts_detrend = ts - ts_mean
    ts_var = ts_detrend.var(dim="time", skipna=True)
    return ts_var
