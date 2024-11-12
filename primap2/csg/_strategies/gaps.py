import numpy as np
import pandas as pd
import xarray as xr
from attrs import frozen
from loguru import logger


@frozen
class Gap:
    type: str = None
    # possible types:
    #   'start': start of timeseries boundary (nan, nan, X, X)
    #   'end': end oftimeseries boundary (X, X, nan, nan)
    #   'ggap': gap (X, nan, nan, X)
    left: np.datetime64 = None  # left end of the gap
    right: np.datetime64 = None  # right end of the gap


def get_gaps(ts: xr.DataArray) -> list[Gap]:
    """
    Find nans on the boundaries of the timeseries and gaps in the time-series

    Parameters
    ----------
    ts :
        Time-series to analyze

    Returns
    -------
        list of Gaps

    """
    ts_roll = ts.rolling(time=3, min_periods=1, center=True).sum()
    gaps = []
    # check for left boundary
    if ts[0].isnull():
        right = ts_roll.dropna(dim="time").coords["time"].data[0]
        gaps.append(Gap(type="start", left=ts.coords["time"].data[0], right=right))
        left_bound = ts.dropna(dim="time").coords["time"].data[0]
    else:
        left_bound = ts.coords["time"].data[0]
    # check for right boundary
    if ts[-1].isnull():
        left = ts_roll.dropna(dim="time").coords["time"].data[-1]
        gaps.append(Gap(type="end", left=left, right=ts.coords["time"].data[-1]))
        right_bound = ts.dropna(dim="time").coords["time"].data[-1]
    else:
        right_bound = ts.coords["time"].data[-1]

    # get gaps. constrain ts using left and right boundary
    ts_inner = ts.pr.loc[{"time": slice(left_bound, right_bound)}]
    if any(ts_inner.isnull()):
        # we have gaps, detect them
        # first make a mask for the nan values
        ts_nan = ts_inner.notnull()
        # make a rolling sum with width two and left alignment and take a nan mask
        ts_left_nan = ts_inner.rolling(time=2, min_periods=1, center=False).sum().notnull()
        # shift the sum right for a right alignment rolling sum
        ts_right_nan = ts_left_nan.shift(time=-1, fill_value=True)
        # by xor-ing with the original nan mask we can get the left and the right boundaries
        left_boundary_mask = xr.apply_ufunc(np.logical_xor, ts_nan, ts_left_nan)
        right_boundary_mask = xr.apply_ufunc(np.logical_xor, ts_nan, ts_right_nan)
        right_boundaries = list(ts_inner.coords["time"].data[right_boundary_mask])
        left_boundaries = list(ts_inner.coords["time"].data[left_boundary_mask])

        for i, left_bound in enumerate(left_boundaries):
            gaps.append(Gap(type="gap", left=left_bound, right=right_boundaries[i]))

    return gaps


def calculate_boundary_trend(
    ts: xr.DataArray,
    gap: Gap,
    fit_degree: int = 1,
    trend_length: int = 1,
    trend_length_unit: str = "YS",
    min_trend_points: int = 1,
) -> xr.DataArray | None:
    """
    Replace boundary points by trend values

    Parameters
    ----------
    ts :
        Time-series to calculate trend for
    gap :
        Gap definition
    fit_degree :
        degree of the polynomial to fit to calculate the trend value. 0 for mean value
        and 1 for linear trend make most sense. The higher the order, the higher the
        chance of unexpected results
    trend_length :
        length of the trend in time steps (usually years)
    trend_length_unit :
        Unit for the length of the trend. String passed to the `freq` argument of
        `pd.date_range`. Default is 'YS' (yearly at start of year)
    min_trend_points :
        minimal number of points to calculate the trend. Default is 1, but if the degree
        of the fit polynomial is higher than 1, the minimal number of data points equals
        the degree of the fit polynomial

    Returns
    -------
        Time-series (xr.DataArray) with boundary point replaced by trend. If trend
        calculation is not possible, `None` is returned so the calling strategy can
        raise the StrategyUnableToProcess error.

    """

    if gap.type == "gap":
        # right boundary
        ts = calculate_right_boundary_trend(
            ts,
            boundary=gap.right,
            fit_degree=fit_degree,
            trend_length=trend_length,
            trend_length_unit=trend_length_unit,
            min_trend_points=min_trend_points,
        )
        if ts is None:
            return None
        # left boundary
        ts = calculate_left_boundary_trend(
            ts,
            boundary=gap.left,
            fit_degree=fit_degree,
            trend_length=trend_length,
            min_trend_points=min_trend_points,
        )
    elif gap.type == "end":
        # left boundary
        ts = calculate_left_boundary_trend(
            ts,
            boundary=gap.right,
            fit_degree=fit_degree,
            trend_length=trend_length,
            min_trend_points=min_trend_points,
        )
    elif gap.type == "start":
        # right boundary
        ts = calculate_right_boundary_trend(
            ts,
            boundary=gap.left,
            fit_degree=fit_degree,
            trend_length=trend_length,
            min_trend_points=min_trend_points,
        )
    else:
        raise ValueError(f"Unknown gap type: {gap.type}")

    return ts


def calculate_right_boundary_trend(
    ts: xr.DataArray,
    boundary: np.datetime64,
    fit_degree: int = 1,
    trend_length: int = 1,
    trend_length_unit: str = "YS",
    min_trend_points: int = 1,
) -> xr.DataArray | None:
    """
    Replace right boundary point by trend value

    The function assumes equally spaced

    Parameters
    ----------
    ts :
        Time-series to calculate trend for
    boundary :
        boundary point (last NaN value)
    fit_degree :
        degree of the polynomial to fit to calculate the trend value. 0 for mean value
        and 1 for linear trend make most sense. The higher the order, the higher the
        chance of unexpected results
    trend_length :
        length of the trend in time steps (usually years)
    trend_length_unit :
        Unit for the length of the trend. String passed to the `freq` argument of
        `pd.date_range`. Default is 'YS' (yearly at start of year)
    min_trend_points :
        minimal number of points to calculate the trend. Default is 1, but if the degree
        of the fit polynomial is higher than 1, the minimal number of data points equals
        the degree of the fit polynomial

    Returns
    -------
        Time-series (xr.DataArray) with boundary point replaced by trend. If trend
        calculation is not possible, `None` is returned so the calling strategy can
        raise the StrategyUnableToProcess error.

    """
    if min_trend_points < fit_degree:
        min_trend_points = fit_degree

    point_to_modify = get_shifted_time_value(ts, original_value=boundary, shift=1)
    ts_fit = ts.pr.loc[
        {"time": pd.date_range(start=point_to_modify, periods=trend_length, freq=trend_length_unit)}
    ]

    if len(ts_fit.where(ts_fit.notnull(), drop=True)) >= min_trend_points:
        fit = ts_fit.polyfit(dim="time", deg=fit_degree, skipna=True)
        value = xr.polyval(
            ts_fit.coords["time"].pr.loc[{"time": point_to_modify}] - ts_fit.coords["time"].data[0],
            fit.polyfit_coefficients,
        )
        ts.pr.loc[{"time": point_to_modify}] = value
        return ts
    else:
        logger.info(
            f"Not enough values to calculate fit for right boundary at "
            f"{point_to_modify}. fit_degree: {fit_degree}, "
            f"trend_length: {trend_length}, "
            f"trend_length_unit: {trend_length_unit}, "
            f"min_trend_points: {min_trend_points}. "
            f"Timeseries info: {timeseries_coord_repr(ts)}"
        )
        return None


def calculate_left_boundary_trend(
    ts: xr.DataArray,
    boundary: np.datetime64,
    fit_degree: int = 1,
    trend_length: int = 1,
    trend_length_unit: str = "YS",
    min_trend_points: int = 1,
) -> xr.DataArray | None:
    """
    Replace left boundary point by trend value

    The function assumes equally spaced

    Parameters
    ----------
    ts :
        Time-series to calculate trend for
    boundary :
        boundary point (last NaN value)
    fit_degree :
        degree of the polynomial to fit to calculate the trend value. 0 for mean value
        and 1 for linear trend make most sense. The higher the order, the higher the
        chance of unexpected results
    trend_length :
        length of the trend in time steps (usually years)
    trend_length_unit :
        Unit for the length of the trend. String passed to the `freq` argument of
        `pd.date_range`. Default is 'YS' (yearly at start of year)
    min_trend_points :
        minimal number of points to calculate the trend. Default is 1, but if the degree
        of the fit polynomial is higher than 1, the minimal number of data points equals
        the degree of the fit polynomial

    Returns
    -------
        Time-series (xr.DataArray) with boundary point replaced by trend. If trend
        calculation is not possible, `None` is returned so the calling strategy can
        raise the StrategyUnableToProcess error.

    """
    if min_trend_points < fit_degree:
        min_trend_points = fit_degree

    point_to_modify = get_shifted_time_value(ts, original_value=boundary, shift=-1)
    ts_fit = ts.pr.loc[
        {"time": pd.date_range(end=point_to_modify, periods=trend_length, freq=trend_length_unit)}
    ]

    if len(ts_fit.where(ts_fit.notnull(), drop=True)) >= min_trend_points:
        fit = ts_fit.polyfit(dim="time", deg=fit_degree, skipna=True)
        value = xr.polyval(
            ts_fit.coords["time"].pr.loc[{"time": point_to_modify}] - ts_fit.coords["time"].data[0],
            fit.polyfit_coefficients,
        )
        ts.pr.loc[{"time": point_to_modify}] = value
        return ts
    else:
        logger.info(
            f"Not enough values to calculate fit for left boundary at "
            f"{point_to_modify}. fit_degree: {fit_degree}, "
            f"trend_length: {trend_length}, "
            f"trend_length_unit: {trend_length_unit}, "
            f"min_trend_points: {min_trend_points}. "
            f"Timeseries info: {timeseries_coord_repr(ts)}"
        )
        return None


def get_shifted_time_value(
    ts: xr.DataArray,
    original_value: np.datetime64,
    shift: int,
) -> np.datetime64:
    """Get time coordinate value `shift` positions from `original_value`

    Parameters
    ----------
    ts :
        time series to use (for time coordinate)
    shift :
        position of the value to get relative to the `original_value`

    Returns
    -------
        time coordinate value at desired relative position

    """
    # TODO: the following is not very elegant. I struggle with tasks like getting the coordinate
    #  value of the next item in xarray
    mask = ts.copy()
    mask.data = mask.data * np.nan
    mask.pr.loc[{"time": original_value}] = 1
    mask = mask.shift(time=shift, fill_value=np.nan)
    return mask.coords["time"].where(mask == 1, drop=True).data[0]


def timeseries_coord_repr(ts: xr.DataArray) -> str:
    """Make short string representation for coordinate values for logging"""
    dims = set(ts.coords._names) - {"time"}
    coords: dict[str, str] = {str(k): ts[k].item() for k in dims}
    coords = dict(sorted(coords.items()))
    return repr(coords)
