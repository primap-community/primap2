"""Merge arrays and datasets with optional tolerances."""

import contextlib
from datetime import date

import numpy as np
import xarray as xr
from loguru import logger

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor


def merge_with_tolerance_core(
    *,
    da_start: xr.DataArray,
    da_merge: xr.DataArray,
    tolerance: float = 0.01,
    error_on_discrepancy: bool = True,
) -> xr.DataArray:
    """
    Merge two DataArrays with a given tolerance for discrepancies in values
    present in both DataArrays.

    If values from the data to merge are already present in da_start they are
    treated as equal if the relative difference is below the tolerance threshold.
    The result will use the values present in da_start.

    If a merge using xr.merge is not possible the function identifies the conflicting
    values and checks if the relative difference is above the tolerance threshold.
    If it is, an error is raised (if error_on_discrepancy = True) or a warning is
    logged (if error_on_discrepancy = False).

    The function assumes dataarrays have been checked for identical coordinates
    etc. as it is called by DataArray.pr.merge() and Dataset.pr.merge().

    Parameters
    ----------
    da_start: xr.DataArray
        data to merge into
    da_merge: xr.DataArray
        data to merge into ds_start
    tolerance: float (optional), default = 0.01
        The tolerance to use when comparing data. Tolerance is relative to values in
        the calling Dataset. Thus by default a 1% deviation of values in da_merge
        from the calling Dataset is tolerated.
    error_on_discrepancy: (optional), default = True
        If true throw an exception if false a warning and return values from
        the calling object in cases of conflict.

    Returns
    -------
        xr.DataArray: DataArray with data from da_merge merged into da_start
    """
    with contextlib.suppress(xr.MergeError):
        da_result = xr.merge(
            [da_start, da_merge],
            compat="no_conflicts",
            join="outer",
        )
        # all done as no errors occurred and thus no duplicates were present
        # make sure we have a DataArray not a Dataset
        return da_result[da_start.name]
    # there are conflicts (overlapping coordinates) between da_start and da_merge

    # calculate the deviation between da_start and da_merge
    da_comp = abs(da_start - da_merge) / da_start
    da_error = da_comp.where(da_comp > tolerance, drop=True)

    if np.logical_not(da_error.isnull()).any():
        # there are differences larger than the tolerance
        log_message = generate_log_message(da_error=da_error, tolerance=tolerance)
        if error_on_discrepancy:
            logger.error(log_message)
            raise xr.MergeError(log_message)
        else:
            # log warning, continue with merging
            logger.warning(log_message)

    # all differences are within the tolerance or ignored, take the first
    # value everywhere
    return da_start.pr.combine_first(da_merge)


def generate_log_message(da_error: xr.DataArray, tolerance: float) -> str:
    """Generate a single large log message for all given errors.

    Strategy:
    * remove all length-1 dimensions and put them in description
    * convert the rest into a pandas dataframe for nice printing
    """
    scalar_dims = [dim for dim in da_error.dims if len(da_error[dim]) == 1]
    scalar_dims_format = []
    for dim in scalar_dims:
        if dim == "time":
            single_date = date.fromtimestamp(da_error[dim].item() / 1000000000)
            scalar_dims_format.append(f"{dim}={single_date.strftime('%Y')}")
        else:
            scalar_dims_format.append(f"{dim}={da_error[dim].item()}")
    scalar_dims_str = ", ".join(scalar_dims_format)
    da_error_dequ = da_error.squeeze(drop=True).pint.dequantify()
    if np.ndim(da_error_dequ.data) == 0:
        errors_str = str(da_error_dequ.data)
    else:
        errors_str = da_error_dequ.to_dataframe().dropna().to_string()

    return (
        f"pr.merge error: found discrepancies larger than tolerance "
        f"({tolerance * 100:.2f}%) for {scalar_dims_str}:\n"
        f"shown are relative discrepancies. ({da_error.name})\n" + errors_str
    )


def ensure_compatible_coords_dims(
    a: xr.Dataset | xr.DataArray, b: xr.Dataset | xr.DataArray
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
        tolerance: float = 0.01,
        error_on_discrepancy: bool = True,
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
            The tolerance to use when comparing data. Tolerance is relative to values in
            the calling Dataset. Thus by default a 1% deviation of values in da_merge
            from the calling Dataset is tolerated.
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
        tolerance: float = 0.01,
        error_on_discrepancy: bool = True,
        combine_attrs: xr.core.types.CombineAttrsOptions = "drop_conflicts",
    ) -> xr.Dataset:
        """
        Merge two Datasets with a given tolerance for discrepancies in values
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
            the calling Dataset. Thus, by default a 1% deviation of values in da_merge
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
        if self._ds.pr.has_processing_info():
            raise NotImplementedError(
                "Dataset contains processing information, this is not supported yet. "
                "Use ds.pr.remove_processing_info()."
            )

        ds_start = self._ds

        with contextlib.suppress(xr.MergeError, ValueError):
            # if there are no conflicts just merge using xr.merge
            return xr.merge(
                [ds_start, ds_merge],
                compat="no_conflicts",
                join="outer",
                combine_attrs=combine_attrs,
            )
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
