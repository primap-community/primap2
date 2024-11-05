from datetime import datetime

import numpy as np
import xarray as xr
from attrs import frozen


@frozen
class Gap:
    type: str = None
    # possible types:
    #   'l': left boundary
    #   'r': right boundary
    #   'g': gap
    left: datetime = None  # left end of the gap
    right: datetime = None  # right end of the gap


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
        gaps.append(Gap(type="l", left=ts.coords["time"].data[0], right=right))
        left_bound = ts.dropna(dim="time").coords["time"].data[0]
    else:
        left_bound = ts.coords["time"].data[0]
    # check for right boundary
    if ts[-1].isnull():
        left = ts_roll.dropna(dim="time").coords["time"].data[-1]
        gaps.append(Gap(type="r", left=left, right=ts.coords["time"].data[-1]))
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
            gaps.append(Gap(type="g", left=left_bound, right=right_boundaries[i]))

    return gaps
