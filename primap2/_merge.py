from typing import Optional

import xarray as xr
from loguru import logger

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor


# general functions
def merge_with_tolerance_core(
    da_start: xr.DataArray,
    da_merge: xr.DataArray,
    tolerance: Optional[float] = 0.01,
    error_on_discrepancy: Optional[bool] = True,
) -> xr.DataArray:
    """
    Merge two DataArrays with a given tolerance for descrepancies in values
    present in both DataArrays.

    If values from the data to merge are already present in da_start they are
    treated as equal if the relative difference is below the tolerance threshold.
    The result will use the values present in da_start.

    If a merge using xr.merge is not possible the function identifies the conflicting
    values using the following recursive algorithm:

    * The coordinates which are non-atomic for the combined DataArrays are identified
    * If non-atomic coordinates exist:
        * The DataArray is split along the first such coordinate.
            * First data for coordinate values only present in one of the DataArrays
            is combined as it can't create conflicts.
            * Then this function is called for the sub-DataArrays for the values present
            in both DataArrays
        * Results from the sub-DataArrays are combined using xr.merge as they can't
        have conflicts any more.
    * if no such coordinate exists we have a single time series for the same
    coordinate values in both DataArrays. The relative deviation is calculated and
    compared to the threshold. If it exceeds the threshold for at least one point in
    time this is logged and if desired an error is thrown (default behaviour)

    The function assumes dataarrays have been checked for identical coordinates
    etc as it is called by DataArray.pr.merge() and Dataset.pr.merge() (and itself).


    Parameters
    ----------
    da_start: xr.DataArray
        data to merge into
    da_merge: xr.DataArray
        data to merge into ds_start
    tolerance: float (optional), default = 0.01
        The tolerance to use when comparing data
    error_on_discrepancy (optional), default = True
        If true throw an exception if false a warning and return values from
        the calling object in cases of conflict.
    combine_attrs (optional), default = "drop_conflicts"
        Governs how to combine conflicting attrs. Is passed on to the xr merge
        functions.

    Returns
    -------
        xr.DataArray: DataArray with data from da_merge merged into the calling object

    """

    entity = da_start.attrs["entity"]
    if "gwp_context" in da_start.attrs:
        entity = f"{entity} ({da_start.attrs['gwp_context']})"
    # try to merge
    try:
        da_result = xr.merge(
            [da_start, da_merge],
            compat="no_conflicts",
            join="outer",
        )
        # make sure we have a DataArray not a Dataset
        da_result = da_result[entity]
        # all done as no errors occurred and thus no duplicates were present
    except xr.MergeError:
        logger.debug("Doing a merge by coordinates")
        # we have conflicting data and try to merge by splitting the DataArray
        # into pieces along coordinate axes and merge for those to isolate the error

        # find non-unique coords to iterate over
        coords_start = da_start.coords
        coords_merge = da_merge.coords
        coords_to_iterate = {}
        for coord in coords_start:
            vals_start = set(coords_start[coord].values)
            vals_merge = set(coords_merge[coord].values)
            all_values = vals_start | vals_merge

            if len(all_values) > 1 and coord != "time":
                # coords_to_iterate.append(coord)
                coords_to_iterate[coord] = {
                    "vals_start": vals_start,
                    "vals_merge": vals_merge,
                }

        if len(coords_to_iterate.keys()) > 0:
            coord = list(coords_to_iterate.keys())[0]
            # determine values in both dataarrays
            # vals_start = set(coords_start[coord].values)
            # vals_merge = set(coords_merge[coord].values)
            vals_start = coords_to_iterate[coord]["vals_start"]
            vals_merge = coords_to_iterate[coord]["vals_merge"]
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
                da_result = da_result[entity]
            elif vals_only_start:
                da_result = da_start.pr.loc[{coord: list(vals_only_start)}]
            elif vals_only_merge:
                da_result = da_merge.pr.loc[{coord: list(vals_only_merge)}]
            else:
                da_result = None

            # or the common values go one by one and merge
            # using the same function recursively
            for value in vals_common:
                # call function recursively
                da_result_new = merge_with_tolerance_core(
                    da_start.pr.loc[{coord: [value]}],
                    da_merge.pr.loc[{coord: [value]}],
                    tolerance=tolerance,
                    error_on_discrepancy=error_on_discrepancy,
                )
                # merge with existing data should not throw error as
                # coordinate values are different
                if da_result is None:
                    da_result = da_result_new
                else:
                    # merge.
                    da_result = xr.merge(
                        [da_result, da_result_new], compat="no_conflicts", join="outer"
                    )
                    # make sure we have a dataarray not a dataset
                    da_result = da_result[entity]
        else:
            # we've reached a single time series so compare by time step
            # dequantify as we're transforming to pandas dataframe
            da_start_dequ = da_start.pr.dequantify()
            unit = da_start_dequ.attrs["units"]
            da_merge_dequ = da_merge.pint.to(unit).pr.dequantify()
            df_start = da_start_dequ.to_dataframe()
            df_merge = da_merge_dequ.to_dataframe()
            index_common = df_start.index.intersection(df_merge.index)
            df_start.loc[index_common].compare(df_merge.loc[index_common])
            # calculate the deviation of data to merge from data start in
            # % of data start values
            df_comp = abs(
                (
                    df_start[entity].loc[index_common]
                    - df_merge[entity].loc[index_common]
                )
                / df_start[entity].loc[index_common]
            )
            # select the values exceeding the tolerance
            df_error = df_comp[df_comp > tolerance]
            if len(df_error) > 0:
                # to include all discrepancies in one message we loop over them
                error_str = ""
                for time in df_error.index.get_level_values("time"):
                    idx = df_error.index.get_level_values("time") == time
                    error_str += (
                        f": {time.strftime('%d-%m-%Y')}: "
                        f"{float(df_error.loc[idx])*100:.2f}%"
                    )
                coords = set(da_start.coords) - {"time"}
                vals = [da_start.coords[coord].data[0] for coord in coords]
                coords_vals = [coord + ": " + val for coord, val in zip(coords, vals)]
                coords_str = ", ".join(coords_vals)
                log_message = (
                    f"pr.merge error: found discrepancies larger than tolerance "
                    f"({tolerance*100:.2f}%) for {coords_str} and times: "
                    f"{error_str}"
                )
                if error_on_discrepancy:
                    logger.error(log_message)
                    raise xr.MergeError(log_message)
                else:
                    logger.warning(log_message)
                    da_result = xr.merge([da_start, da_merge], compat="override")
            else:
                # take first value as they are the same given the tolerance
                da_result = xr.merge([da_start, da_merge], compat="override")
    return da_result


class DataArrayMergeAccessor(BaseDataArrayAccessor):
    def merge(
        self,
        da_merge: xr.DataArray,
        tolerance: Optional[float] = 0.01,
        error_on_discrepancy: Optional[bool] = True,
    ) -> xr.DataArray:
        """
        Merge this data array with another using a given tolerance for
        descrepancies in values present in both DataArrays.

        If values from the data to merge are already
        present they are treated as equal if the relative
        difference is below the tolerance threshold.

        Parameters
        ----------
        da_merge: xr.DataArray
            data to merge to the calling object
        tolerance: float (optional), default = 0.01
            The tolerance to use when comparing data
        error_on_discrepancy (optional), default = True
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
        coords_start = set(da_start.coords)
        coords_merge = set(da_merge.coords)
        if coords_start != coords_merge:
            # ToDo: custom error
            logger.error("pr.merge error: coords of dataarrays to merge must agree")
            raise ValueError("pr.merge error: coords of dataarrays to merge must agree")

        dims_start = da_start.dims
        dims_merge = da_merge.dims
        if dims_start != dims_merge:
            # ToDo: custom error
            logger.error("pr.merge error: dims of dataarrays to merge must agree")
            raise ValueError("pr/merge error: dims of dataarrays to merge must agree")

        da_result = merge_with_tolerance_core(
            da_start,
            da_merge,
            tolerance=tolerance,
            error_on_discrepancy=error_on_discrepancy,
        )

        return da_result


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
        da_merge: xr.Dataset
            data to merge to the calling object
        tolerance: float (optional), default = 0.01
            The tolerance to use when comparing data. Tolerance is relative to values in
            the calling Dataset. Thus by default a 1% deviation of values in da_merge
            from the calling Dataset is tolerated.
        error_on_discrepancy (optional), default = True
            If true throw an exception if false a warning and return values from
            the calling object in cases of conflict.
        combine_attrs (optional), default = "drop_conflicts"
            Governs how to combine conflicting attrs. Is passed on to the xr merge
            functions.

        Returns
        -------
            xr.Dataset: Dataset with data from da_merge merged into the calling object
        """
        ds_start = self._ds

        try:
            # if there are no conflicts just merge using xr.merge
            ds_result = xr.merge(
                [ds_start, ds_merge],
                compat="no_conflicts",
                join="outer",
                combine_attrs=combine_attrs,
            )
        except xr.MergeError:
            # merge by hand
            # check if coordinates and dimensions agree
            coords_start = set(ds_start.coords)
            coords_merge = set(ds_merge.coords)
            if coords_start != coords_merge:
                logger.error("pr.merge error: coords of datasets to merge must agree")
                raise ValueError(
                    "pr.merge error: coords of datasets to merge must agree"
                )

            dims_start = set(ds_start.dims)
            dims_merge = set(ds_merge.dims)
            if dims_start != dims_merge:
                logger.error("pr.merge error: dims of datasets to merge must agree")
                raise ValueError("pr.merge error: dims of datasets to merge must agree")

            vars_start = set(ds_start.data_vars)
            vars_merge = set(ds_merge.data_vars)
            vars_common = vars_start & vars_merge
            vars_only_start = vars_start - vars_common
            vars_only_merge = vars_merge - vars_common

            ds_result = xr.merge(
                [ds_start[vars_only_start], ds_merge[vars_only_merge]],
                combine_attrs=combine_attrs,
            )

            for var in vars_common:
                print(f"merging for {var}")
                ds_result_new = merge_with_tolerance_core(
                    ds_start[var],
                    ds_merge[var],
                    tolerance=tolerance,
                    error_on_discrepancy=error_on_discrepancy,
                )
                ds_result = xr.merge(
                    [ds_result, ds_result_new], combine_attrs="override"
                )
        return ds_result
