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
    combine_attrs: Optional[str] = "drop_conflicts",
) -> xr.DataArray:
    """
    Merge two DataArrays with a given tolerance for descrepancies in values
    present in both DataArrays. If values from the data to merge are already
    present in da_start they are treated as equal if the relative difference
    is below the tolerance threshold. The result will use the values present
     in da_start.

    The function determines non-unique coordinates and iterates over them to
    try merging for individual values of the coordinates. Where merge for a value
    fails because of unequal data, the function recursively dives into the other
    coordinates until single time-series are left which are then analyzed for each
    time step and errors are logged as desired.

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
    # try to merge
    try:
        da_result = xr.merge(
            [da_start, da_merge],
            compat="no_conflicts",
            join="outer",
            combine_attrs=combine_attrs,
        )
        # make sure we have a DataArray not a Dataset
        da_result = da_result[entity]
        # all done as no errors occurred and thus no duplicates were present
    except xr.MergeError:
        # print(f"Conflicting data: {err}")
        # print("Doing a merge by coordinates")
        # we have conflicting data and try to merge by splitting the DataArray
        # into pieces along coordinate axes and merge for those to isolate the error

        # find non-unique coords to iterate over
        coords_start = da_start.coords
        coords_merge = da_merge.coords
        coords_to_iterate = []
        for coord in list(coords_start):
            vals_start = set(coords_start[coord].values)
            vals_merge = set(coords_merge[coord].values)
            all_values = vals_start | vals_merge

            if len(all_values) > 1 and coord != "time":
                coords_to_iterate.append(coord)
        # print(f"Found non-unique coordinates {coords_to_iterate}.")

        if len(coords_to_iterate) > 0:
            coord = coords_to_iterate[0]
            # determine value sin both dataarrays
            vals_start = set(coords_start[coord].values)
            vals_merge = set(coords_merge[coord].values)
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
            # calculate the deviation od data to merge from data start in
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
                error_str = None
                for time in df_error.index.get_level_values("time"):
                    idx = df_error.index.get_level_values("time") == time
                    if error_str is None:
                        error_str = (
                            f"{time.strftime('%d-%m-%Y')}: "
                            f"{float(df_error.loc[idx])*100:.2f}%"
                        )
                    else:
                        error_str = (
                            error_str + f": {time.strftime('%d-%m-%Y')}: "
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
        combine_attrs: Optional[str] = "drop_conflicts",
    ) -> xr.DataArray:
        """
        Merge two DataArrays with a given tolerance for descrepancies in values
        present in both DataArrays. If values from the data to merge are already
        present in the calling object they are treated as equal if the relative
        difference is below the tolerance threshold. The result will use the values
        of the calling object.

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
            combine_attrs=combine_attrs,
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
            The tolerance to use when comparing data
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
        # check if coordinates and dimensions agree
        coords_start = set(ds_start.coords)
        coords_merge = set(ds_merge.coords)
        if coords_start != coords_merge:
            logger.error("pr.merge error: coords of datasets to merge must agree")
            raise ValueError("pr.merge error: coords of datasets to merge must agree")

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
        print(f"vars_start: {vars_start}, ({vars_only_start})")
        print(f"vars_merge: {vars_merge}, ({vars_only_merge})")

        if (len(vars_only_start) > 0) & (len(vars_only_merge) == 0):
            print("start with start")
            ds_result = ds_start[vars_only_start]
        elif (len(vars_only_start) == 0) & (len(vars_only_merge) > 0):
            print("start with merge")
            ds_result = ds_merge[vars_only_merge]
        elif (len(vars_only_start) == 0) & (len(vars_only_merge) == 0):
            # use df_start as starting point. All variables will be overwritten
            # but we have a non-empty dataset structure to fill with the
            # DataArrays
            print("start with ds_start to overwrite")
            ds_result = ds_start.copy(deep=True)
        else:
            print(
                f"vars_only_start: {vars_only_start}, vars_only_merge:"
                f" {vars_only_merge}"
            )
            ds_result = xr.merge([ds_start[vars_only_start], ds_merge[vars_only_merge]])

        for var in vars_common:
            ds_result[var] = merge_with_tolerance_core(
                ds_start[var],
                ds_merge[var],
                tolerance=tolerance,
                error_on_discrepancy=error_on_discrepancy,
                combine_attrs=combine_attrs,
            )

        return ds_result
