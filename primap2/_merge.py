"""Merge arrays and datasets with optional tolerances."""

import typing
from typing import Optional

import xarray as xr
from loguru import logger

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor


def merge_with_tolerance_core(
    *,
    da_start: xr.DataArray,
    da_merge: xr.DataArray,
    tolerance: Optional[float] = 0.01,
    error_on_discrepancy: Optional[bool] = True,
) -> xr.DataArray:
    """
    Merge two DataArrays with a given tolerance for discrepancies in values
    present in both DataArrays.

    If values from the data to merge are already present in da_start they are
    treated as equal if the relative difference is below the tolerance threshold.
    The result will use the values present in da_start.

    If a merge using xr.merge is not possible the function identifies the conflicting
    values using the following recursive algorithm:

    * The coordinates which are non-atomic for the combined DataArrays are identified
    * If non-atomic coordinates exist:
        - The DataArray is split along the first such coordinate.
        - First data for coordinate values only present in one of the DataArrays
          is combined as it can't create conflicts.
        - Then this function is called for the sub-DataArrays for the values present
          in both DataArrays
        - Results from the sub-DataArrays are combined using xr.merge as they can't
          have conflicts any more.
    * if no such coordinate exists we have a single time series for the same
      coordinate values in both DataArrays. The relative deviation is calculated and
      compared to the threshold. If it exceeds the threshold for at least one point in
      time this is logged and if desired an error is thrown (default behaviour)

    The function assumes dataarrays have been checked for identical coordinates
    etc. as it is called by DataArray.pr.merge() and Dataset.pr.merge() (and itself).

    TODO: this is slow with higher-dimensional data. Probably, it isn't necessary
    to drill down to time series to collect discrepancies?

    Parameters
    ----------
    da_start: xr.DataArray
        data to merge into
    da_merge: xr.DataArray
        data to merge into ds_start
    tolerance: float (optional), default = 0.01
        The tolerance to use when comparing data
    error_on_discrepancy: (optional), default = True
        If true throw an exception if false a warning and return values from
        the calling object in cases of conflict.

    Returns
    -------
        xr.DataArray: DataArray with data from da_merge merged into da_start
    """

    try:
        da_result = xr.merge(
            [da_start, da_merge],
            compat="no_conflicts",
            join="outer",
        )
        # all done as no errors occurred and thus no duplicates were present
        # make sure we have a DataArray not a Dataset
        return da_result[da_start.name]
    except xr.MergeError:
        pass

    logger.debug("Doing a merge by coordinates")
    # we have conflicting data and try to merge by splitting the DataArray
    # into pieces along coordinate axes and merge for those to isolate the error

    # find if any coord still needs to be treated, i.e. iterated over
    for coord in da_start.coords:
        if coord == "time":
            continue
        vals_start = set(da_start.coords[coord].values)
        vals_merge = set(da_merge.coords[coord].values)
        all_values = vals_start | vals_merge

        if len(all_values) > 1:
            # merge_array_by_coord will recursively call merge_with_tolerance_core
            # if discrepancies along other dimensions are encountered.
            return merge_array_by_coord(
                coord=coord,
                vals_start=vals_start,
                vals_merge=vals_merge,
                da_start=da_start,
                da_merge=da_merge,
                error_on_discrepancy=error_on_discrepancy,
                tolerance=tolerance,
            )

    # did NOT find any coord which needs to be treated
    # that means we are merging a single time series
    return merge_timeseries(
        da_start=da_start,
        da_merge=da_merge,
        error_on_discrepancy=error_on_discrepancy,
        tolerance=tolerance,
    )


def merge_timeseries(
    *,
    da_start: xr.DataArray,
    da_merge: xr.DataArray,
    error_on_discrepancy: typing.Union[None, bool],
    tolerance: typing.Union[None, float],
) -> xr.DataArray:
    """Merge two timeseries (dataArrays with one non-scalar dimension, the time)."""
    logger.debug("Merging time series")

    # calculate the deviation between da_start and da_merge
    da_comp = abs(da_start - da_merge) / da_start
    da_error = da_comp.where(da_comp > tolerance, drop=True)

    if len(da_error) > 0:
        # include all discrepancies in one message
        log_message = generate_log_message(
            da_error=da_error, da_start=da_start, tolerance=tolerance
        )
        if error_on_discrepancy:
            logger.error(log_message)
            raise xr.MergeError(log_message)
        else:
            # log warning, continue with merging
            logger.warning(log_message)

    # take first value as they are the same given the tolerance
    return xr.merge([da_start, da_merge], compat="override")[da_start.name]


def generate_log_message(
    da_error: xr.DataArray, da_start: xr.DataArray, tolerance: float
) -> str:
    """Generate a single large log message for all given errors."""
    da_times = da_error.time.dt.strftime("%Y-%m-%d")
    error_str = "\n".join(
        f"{da_times.loc[time].item()}: "
        f"{da_error.loc[time].item().magnitude * 100:.2f}%"
        for time in da_error.time
    )

    coords = set(da_start.coords) - {"time"}
    vals = [da_start[coord].item() for coord in coords]
    coords_str = ", ".join(f"{coord}: {val}" for coord, val in zip(coords, vals))
    log_message = (
        "pr.merge error: found discrepancies larger than tolerance "
        f"({tolerance * 100:.2f}%) for {coords_str} and times:\n"
        f"{error_str}"
    )
    return log_message


def merge_array_by_coord(
    *,
    coord: typing.Hashable,
    da_start: xr.DataArray,
    da_merge: xr.DataArray,
    vals_start: set,
    vals_merge: set,
    error_on_discrepancy: typing.Union[None, bool],
    tolerance: typing.Union[None, float],
) -> xr.DataArray:
    """Merge two DataArrays along the given coordinate, resolving conflicts one by
    one."""
    logger.debug(f"Merging along {coord}")
    # values which are in both dataArrays
    vals_common = vals_start & vals_merge
    # First treat the values only present in one of the datasets
    # as these can't create merge conflicts
    vals_only_start = vals_start - vals_common
    vals_only_merge = vals_merge - vals_common
    if vals_only_start and vals_only_merge:
        da_result = xr.merge(
            [
                da_start.pr.loc[{coord: list(vals_only_start)}],
                da_merge.pr.loc[{coord: list(vals_only_merge)}],
            ],
            compat="no_conflicts",
            join="outer",
        )
        # make sure we have a dataarray not a dataset
        da_result = da_result[da_start.name]
    elif vals_only_start:
        da_result = da_start.pr.loc[{coord: list(vals_only_start)}]
    elif vals_only_merge:
        da_result = da_merge.pr.loc[{coord: list(vals_only_merge)}]
    else:
        da_result = None

    # the common values go one by one and are merged using the merge function
    # recursively
    for value in vals_common:
        da_conflicting = merge_with_tolerance_core(
            da_start=da_start.pr.loc[{coord: [value]}],
            da_merge=da_merge.pr.loc[{coord: [value]}],
            tolerance=tolerance,
            error_on_discrepancy=error_on_discrepancy,
        )
        if da_result is None:
            da_result = da_conflicting
        else:
            da_result = xr.merge(
                [da_result, da_conflicting], compat="no_conflicts", join="outer"
            )

    # make sure we have a dataarray not a dataset
    return da_result[da_start.name]


def ensure_compatible_coords_dims(
    a: typing.Union[xr.Dataset, xr.DataArray], b: typing.Union[xr.Dataset, xr.DataArray]
):
    """Check if coordinates and dimensions of both Datasets or DataArrays agree,
    raise exception otherwise."""
    if set(a.coords) != set(b.coords):
        logger.error("pr.merge error: coords of objects to merge must agree")
        raise ValueError("pr.merge error: coords of objects to merge must agree")
    if set(a.dims) != set(b.dims):
        logger.error("pr.merge error: dims of objects to merge must agree")
        raise ValueError("pr.merge error: dims of objects to merge must agree")


class DataArrayMergeAccessor(BaseDataArrayAccessor):
    def merge(
        self,
        da_merge: xr.DataArray,
        tolerance: Optional[float] = 0.01,
        error_on_discrepancy: Optional[bool] = True,
    ) -> xr.DataArray:
        """
        Merge this data array with another using a given tolerance for
        discrepancies in values present in both DataArrays.

        If values from the data to merge are already
        present they are treated as equal if the relative
        difference is below the tolerance threshold.

        Parameters
        ----------
        da_merge: xr.DataArray
            data to merge to the calling object
        tolerance: float (optional), default = 0.01
            The tolerance to use when comparing data
        error_on_discrepancy: (optional), default = True
            If true throw an exception if false a warning and return values from
            the calling object in cases of conflict.
        combine_attrs (optional), default = "drop_conflicts"
            Governs how to combine conflicting attrs. Is passed on to the xr merge
            functions.

        Returns
        -------
            xr.DataArray: DataArray with data from da_merge merged into the calling
            object
        """

        # check if coordinates and dimensions agree
        da_start = self._da
        ensure_compatible_coords_dims(da_start, da_merge)

        return merge_with_tolerance_core(
            da_start=da_start,
            da_merge=da_merge,
            tolerance=tolerance,
            error_on_discrepancy=error_on_discrepancy,
        )


class DatasetMergeAccessor(BaseDatasetAccessor):
    def merge(
        self,
        ds_merge: xr.Dataset,
        tolerance: Optional[float] = 0.01,
        error_on_discrepancy: Optional[bool] = True,
        combine_attrs: Optional[str] = "drop_conflicts",
    ) -> xr.Dataset:
        """
        Merge two Datasets with a given tolerance for descrepancies in values
        present in both Datasets. If values from the data to merge are already
        present in the calling object they are treated as equal if the relative
        difference is below the tolerance threshold. The result will use the values
        of the calling object.

        Parameters
        ----------
        ds_merge: xr.Dataset
            data to merge to the calling object
        tolerance: float (optional), default = 0.01
            The tolerance to use when comparing data. Tolerance is relative to values in
            the calling Dataset. Thus by default a 1% deviation of values in da_merge
            from the calling Dataset is tolerated.
        error_on_discrepancy: (optional), default = True
            If true throw an exception if false a warning and return values from
            the calling object in cases of conflict.
        combine_attrs: (optional), default = "drop_conflicts"
            Governs how to combine conflicting attrs. Is passed on to the xr merge
            functions.

        Returns
        -------
            xr.Dataset: Dataset with data from da_merge merged into the calling object
        """
        ds_start = self._ds

        try:
            # if there are no conflicts just merge using xr.merge
            return xr.merge(
                [ds_start, ds_merge],
                compat="no_conflicts",
                join="outer",
                combine_attrs=combine_attrs,
            )
        except (xr.MergeError, ValueError):
            pass

        # merge by hand
        ensure_compatible_coords_dims(ds_merge, ds_start)

        vars_start = set(ds_start.data_vars)
        vars_merge = set(ds_merge.data_vars)
        vars_common = vars_start & vars_merge
        vars_only_start = vars_start - vars_common
        vars_only_merge = vars_merge - vars_common

        # merge variables which are only in one dataset and therefore trivially
        # mergeable
        ds_result = xr.merge(
            [ds_start[vars_only_start], ds_merge[vars_only_merge]],
            combine_attrs=combine_attrs,
        )

        # merge potentially problematic variables which are in both datasets
        for var in vars_common:
            logger.debug(f"merging for {var}")
            ds_result_new = merge_with_tolerance_core(
                da_start=ds_start[var],
                da_merge=ds_merge[var],
                tolerance=tolerance,
                error_on_discrepancy=error_on_discrepancy,
            )
            ds_result = xr.merge([ds_result, ds_result_new], combine_attrs="override")
        return ds_result
