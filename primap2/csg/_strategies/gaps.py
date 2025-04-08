import typing

import numpy as np
import pandas as pd
import xarray as xr
from attrs import frozen
from loguru import logger


@frozen
class Gap:
    """
    Class to define a gap in a time-series

    Attributes
    ----------
    type
        type of the gap
        possible types:
            'start': start of timeseries boundary (nan, nan, X, X)
            'end': end of timeseries boundary (X, X, nan, nan)
            'gap': gap (X, nan, nan, X)
    left
        left end of the gap
    right
        right end of the gap
    """

    type: str

    left: np.datetime64  # left end of the gap
    right: np.datetime64  # right end of the gap

    def get_date_slice(self) -> dict[str, slice]:
        """Return a xr.loc type filter for 'time' with a slice from left to right
        end of the gap."""
        return {"time": slice(self.left, self.right)}


@frozen
class FitParameters:
    """
    Class to represent parameters for a polynomial fit.

    While `min_data_points` refers
    to the actual number of data points `trend_length` does not. `trend_length` and
    `trend_length_unit` together define a time span which is independent of the actual
    data points and their spacing.

    Note:
        Very unevenly distributed data points can lead to fit problems,
        e.g. if we use a 10 year period and have 5 data point all in one year. But as the
        normal use case are evenly distributed data points, sometimes with gaps it's
        currently not relevant

    Attributes
    ----------
    fit_degree :
        degree of the polynomial to fit to calculate the trend value. 0 for mean value
        and 1 for linear trend make most sense. The higher the order, the higher the
        chance of unexpected results
    fallback_degree :
        Fallback degree to use if less than min_trend_points of not-nan data
    trend_length :
        length of the trend in time steps (usually years)
    trend_length_unit :
        Unit for the length of the trend. String passed to the `freq` argument of
        `pd.date_range`. Default is 'YS' (yearly at start of year)
    min_trend_points :
        minimal number of points to calculate the trend. Default is 1, but if the degree
        of the fit polynomial is higher than 1, the minimal number of data points
        the degree of the fit polynomial
    """

    fit_degree: int = 1
    fallback_degree: int = 0
    trend_length: int = 10
    trend_length_unit: str = "YS"
    min_trend_points: int = 5

    def __attrs_post_init__(self):
        if self.min_trend_points < self.fit_degree:
            raise ValueError(
                f"min_trend_points ({self.min_trend_points}) "
                f"must not be smaller than "
                f"fit_degree ({self.fit_degree})."
            )

    def log_string(self, fallback: bool = False) -> str:
        """Create a string with the classes parameters."""
        log_str = (
            f"fit_degree: {self.fit_degree}, "
            f"trend_length: {self.trend_length}, "
            f"trend_length_unit: {self.trend_length_unit}, "
            f"min_trend_points: {self.min_trend_points}"
        )
        if fallback:
            log_str = log_str + f", fallback_degree: {self.fallback_degree}.\n"
        else:
            log_str = log_str + ".\n"
        return log_str

    def get_fallback(self):
        """Return FitParameters object with the `fit_degree` set to the `fallback_degree`
        of the original object."""
        return FitParameters(
            fit_degree=self.fallback_degree,
            trend_length=self.trend_length,
            trend_length_unit=self.trend_length_unit,
            min_trend_points=1,
        )


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


def calculate_boundary_trend_with_fallback(
    ts: xr.DataArray,
    gap: Gap,
    fit_params: FitParameters,
) -> np.array:
    """
    Calculate trend values for boundary points. Uses fallback if not enough fit points
    available.

    Parameters
    ----------
    ts :
        Time-series to calculate trend for
    gap :
        Gap definition
    fit_params :
        FitParameters object which holds all parameters for the fit

    Returns
    -------
        Tuple with calculated trend values for left and right boundary of the gap. If trend
        calculation is not possible, `None` is returned so the calling strategy can
        raise the StrategyUnableToProcess error.
    """
    trend_ts = calculate_boundary_trend(
        ts,
        gap=gap,
        fit_params=fit_params,
    )
    if any(np.isnan(trend_ts)):
        trend_ts = calculate_boundary_trend(
            ts,
            gap=gap,
            fit_params=fit_params.get_fallback(),
        )
        if all(np.isnan(trend_ts)):
            logger.info(
                f"Not enough values to calculate fit for ts and gap:"
                f"{gap.type}, [{gap.left}:{gap.right}].\n"
                f"{fit_params.log_string(fallback=True)}"
                f"Timeseries info: {timeseries_coord_repr(ts)}"
            )
        # fill nan from other boundary
        if np.isnan(trend_ts[0]):
            trend_ts[0] = trend_ts[1]
        elif np.isnan(trend_ts[1]):
            trend_ts[1] = trend_ts[0]

    return trend_ts


def calculate_boundary_trend(
    ts: xr.DataArray,
    gap: Gap,
    fit_params: FitParameters,
) -> np.array:
    """
    Calculate trend values for boundary points.

    Parameters
    ----------
    ts :
        Time-series to calculate trend for
    gap :
        Gap definition
    fit_params :
        FitParameters object which holds all parameters for the fit. This function does
        not handle fallback options thus the fallback attribute is ignored.

    Returns
    -------
        Tuple with calculated trend values for left and right boundary of the gap. If trend
        calculation is not possible, `None` is returned so the calling strategy can
        raise the StrategyUnableToProcess error.
    """
    if gap.type == "gap":
        # right boundary
        right = calculate_boundary_trend_inner(
            ts,
            side="right",
            boundary=gap.right,
            fit_params=fit_params,
        )
        # left boundary
        left = calculate_boundary_trend_inner(
            ts,
            side="left",
            boundary=gap.left,
            fit_params=fit_params,
        )
    elif gap.type == "end":
        # left boundary
        left = calculate_boundary_trend_inner(
            ts,
            side="left",
            boundary=gap.left,
            fit_params=fit_params,
        )
        right = left
    elif gap.type == "start":
        # right boundary
        right = calculate_boundary_trend_inner(
            ts,
            side="right",
            boundary=gap.right,
            fit_params=fit_params,
        )
        left = right
    else:
        raise ValueError(f"Unknown gap type: {gap.type}")

    return np.array([left, right])


def calculate_boundary_trend_inner(
    ts: xr.DataArray,
    side: typing.Literal["left", "right"],
    boundary: np.datetime64,
    fit_params: FitParameters,
) -> float:
    """
    Calculate trend value for leftmost or rightmost boundary point.

    Parameters
    ----------
    ts :
        Time-series to calculate trend for
    side : "left" or "right"
        If the left or right boundary point should be processed.
    boundary :
        time index boundary point (last NaN value)
    fit_params :
        FitParameters object which holds all parameters for the fit.
        This function does not handle fallback options thus the fallback
        attribute is ignored.

    Returns
    -------
        Calculated trend value for boundary point. If trend
        calculation is not possible, `None` is returned so the calling strategy can
        raise the StrategyUnableToProcess error.
    """
    point_to_modify = get_shifted_time_value(
        ts, original_value=boundary, shift=1 if side == "right" else -1
    )
    trend_index = pd.date_range(
        start=point_to_modify if side == "right" else None,
        end=point_to_modify if side == "left" else None,
        periods=fit_params.trend_length,
        freq=fit_params.trend_length_unit,
    )
    trend_index = trend_index.intersection(ts.coords["time"])
    ts_fit = ts.pr.loc[{"time": trend_index}]

    if len(ts_fit.where(ts_fit.notnull(), drop=True)) >= fit_params.min_trend_points:
        fit = ts_fit.polyfit(dim="time", deg=fit_params.fit_degree, skipna=True)
        value = xr.polyval(
            ts_fit.coords["time"].pr.loc[{"time": point_to_modify}],
            fit.polyfit_coefficients,
        )
        return float(value.data)
    else:
        logger.info(
            f"Not enough values to calculate fit for {side} boundary at "
            f"{point_to_modify}.\n"
            f"{fit_params.log_string(fallback=False)}"
            f"Timeseries info: {timeseries_coord_repr(ts)}"
        )
        return np.nan


def calculate_scaling_factor(
    ts: xr.DataArray,
    fill_ts: xr.DataArray,
    gap: Gap,
    fit_params: FitParameters,
) -> np.array:
    """
    Calculate scaling factor(s) to fill gaps

    Both trend values (for ts and fill_ts) are calculated in the same way. If default
    trend fails for one of the timeseries we use the fallback for both, so we compare
    the same thing when calculating the factors

    Parameters
    ----------
    ts :
        Timeseries ith gaps to fill
    fill_ts :
        Timeseries to fill gaps with
    gap :
        Definition of the gap
    fit_params :
        FitParameters object which holds all parameters for the fit.

    Returns
    -------
        tuple with left and right scaling factors. For a start and end boundary one of
        the elements is None
    """
    trend_ts = calculate_boundary_trend_with_fallback(
        ts,
        gap=gap,
        fit_params=fit_params,
    )
    if any(np.isnan(trend_ts)):
        # logging has been done already
        return np.array([np.nan, np.nan])

    # trend values for fill_ts
    trend_fill = calculate_boundary_trend_with_fallback(
        fill_ts,
        gap=gap,
        fit_params=fit_params,
    )
    if any(np.isnan(trend_fill)):
        # logging has been done already
        return np.array([np.nan, np.nan])

    factor = np.divide(trend_ts, trend_fill)
    if any(np.isnan(factor)) or any(np.isinf(factor)):
        # we have some nan or inf values which have to come from division by zero
        # we fill them with 0 in case the trend values are zero as well
        nan_mask_factor = np.isnan(factor) | np.isinf(factor)
        zero_mask_ts = trend_ts == 0
        factor[nan_mask_factor & zero_mask_ts] = trend_ts[nan_mask_factor & zero_mask_ts]

    return factor


def fill_gap(
    ts: xr.DataArray,
    fill_ts: xr.DataArray,
    gap: Gap,
    factor: np.array,
) -> xr.DataArray:
    """Fill a gap in a timeseries using data from another timeseries and factors.

    Parameters
    ----------
    ts: xr.DataArray
        Timeseries with a gap to be filled.
    fill_ts: xr.DataArray
        Timeseries from which to take the data to fill in the gap in ts.
    gap: Gap
        The gap that should be filled.
    factor: np.array
        Numpy array containing two floats, the scaling factor to use at the left side of the gap
        and the scaling factor to use at the right side of the gap. The data from fill_ts will be
        multiplied with the factor before filling the gap in ts. If the factors differ, a linear
        interpolation of the scaling factor between both values will be performed.

    Returns
    -----------
        result
            DataArray with the same size and shape as ts, but with the given gap filled
            from fill_ts.
    """
    if factor[0] == factor[1]:
        fill_gap_ts = fill_ts.pr.loc[gap.get_date_slice()] * factor[0]
    else:
        delta_t = gap.right - gap.left
        factor_ts = fill_ts.pr.loc[gap.get_date_slice()].copy()
        factor_ts.data = [
            factor[0] * (gap.right - t) / delta_t + factor[1] * (t - gap.left) / delta_t
            for t in fill_ts.coords["time"].pr.loc[gap.get_date_slice()]
        ]
        fill_gap_ts = fill_ts.pr.loc[gap.get_date_slice()] * factor_ts

    # fill nans (in the gap only)
    ts_aligned, fill_ts_aligned = xr.align(ts, fill_gap_ts, join="left")
    filled_ts = ts_aligned.fillna(fill_ts_aligned)

    return filled_ts


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
    # For actually getting the index of a value in an index, it is easiest to work with
    # the underlying numpy arrays.
    time_points = ts["time"].values
    original_index = np.where(time_points == original_value)[0][0]
    new_index = original_index + shift
    return time_points[new_index]


def timeseries_coord_repr(ts: xr.DataArray) -> str:
    """Make short string representation for coordinate values for logging"""
    dims = set(ts.coords.keys()) - {"time"}
    coords: dict[str, str] = {str(k): ts[k].item() for k in dims}
    coords = dict(sorted(coords.items()))
    return repr(coords)
