"""workaround for xarray coordinate bug

Additional coordinates are lost during operations that require aligning if the
coordinates differ for the structures to be aligned.

see https://github.com/pydata/xarray/issues/9124

"""

import xarray as xr
from loguru import logger

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor


class DataArrayFillAccessor(BaseDataArrayAccessor):
    def fillna(self: xr.DataArray, da_fill: xr.DataArray) -> xr.DataArray:
        """
        Wrapper for `fillna` which ensures that additional coordinates present in
        the calling DataArray are also present in the result. The default xarray
        `fillna` implementation silently drops additional (non-indexed) coordinates
        if they need alignment (do not cover the same values in both DataArrays)


        Parameters
        ----------
        da_fill
            DataArray used to fill nan values in the calling DataArray

        Returns
        -------
            calling DatArray where nan values are filled from da_fill where possible

        """
        da_start = self._da

        coords_start = da_start.coords
        coords_fill = da_fill.coords
        filled = da_start.fillna(da_fill)
        # check if we lost a coordinate
        missing_coords = set(coords_start).difference(set(filled.coords))
        for coord in missing_coords:
            try:
                # merge and align the coordinate
                logger.info(f"adding coord: {coord}")
                coords_merged = coords_start[coord].fillna(coords_fill[coord])
                aligned = xr.align(filled, coords_merged, join="outer")
                # set the coordinate
                filled = aligned[0].assign_coords(
                    {coord: (coords_start[coord].dims, aligned[1].data)}
                )
            except Exception as ex:
                message = f"Could not re-add lost coordinate {coord}: {ex}"
                logger.error(message)
                raise ValueError(message) from ex

        return filled

    def combine_first(self: xr.DataArray, da_combine: xr.DataArray) -> xr.DataArray:
        """
        Wrapper for `combine_first` which ensures that additional coordinates
        present in the calling dataset are also present in the result. The default
        xarray `fillna` implementation silently drops additional (non-indexed)
        coordinates if they need alignment (do not cover the same values in both
        DataArrays)


        Parameters
        ----------
        da_combine
            DataArray used to combine with the calling DataArray

        Returns
        -------
            calling DataArray combined with da_combine

        """
        da_start = self._da

        coords_start = da_start.coords
        coords_fill = da_combine.coords
        filled = da_start.combine_first(da_combine)
        # check if we lost a coordinate
        missing_coords = set(coords_start).difference(set(filled.coords))
        for coord in missing_coords:
            try:
                # merge and align the coordinate
                logger.info(f"adding coord: {coord}")
                coords_merged = coords_start[coord].combine_first(coords_fill[coord])
                aligned = xr.align(filled, coords_merged, join="outer")
                # set the coordinate
                filled = aligned[0].assign_coords(
                    {coord: (coords_start[coord].dims, aligned[1].data)}
                )
            except Exception as ex:
                message = f"Could not re-add lost coordinate {coord}: {ex}"
                logger.error(message)
                raise ValueError(message) from ex

        return filled


class DatasetFillAccessor(BaseDatasetAccessor):
    def fillna(
        self: xr.Dataset,
        ds_fill: xr.Dataset | xr.DataArray,
    ) -> xr.Dataset:
        """
        Wrapper for `fillna` which ensures that additional coordinates present in
        the calling Dataset are also present in the result. The default xarray
        `fillna` implementation silently drops additional (non-indexed) coordinates
        if they need alignment (do not cover the same values in both Datasets)


        Parameters
        ----------
        ds_fill
            Dataset or DataArray used to fill nan values in the calling dataset

        Returns
        -------
            calling Dataset where nan values are filled from ds_fill where possible

        """

        if self._ds.pr.has_processing_info():
            raise NotImplementedError(
                "Dataset contains processing information, this is not supported yet. "
                "Use ds.pr.remove_processing_info()."
            )

        ds_start = self._ds

        coords_start = ds_start.coords
        coords_fill = ds_fill.coords
        filled = ds_start.fillna(ds_fill)
        # check if we lost a coordinate
        missing_coords = set(coords_start).difference(set(filled.coords))
        for coord in missing_coords:
            try:
                # merge and align the coordinate
                logger.info(f"adding coord: {coord}")
                coords_merged = coords_start[coord].fillna(coords_fill[coord])
                aligned = xr.align(filled, coords_merged, join="outer")
                # set the coordinate
                filled = aligned[0].assign_coords(
                    {coord: (coords_start[coord].dims, aligned[1].data)}
                )
            except Exception as ex:
                message = f"Could not re-add lost coordinate {coord}: {ex}"
                logger.error(message)
                raise ValueError(message) from ex

        return filled

    def combine_first(
        self: xr.Dataset,
        ds_combine: xr.Dataset | xr.DataArray,
    ) -> xr.Dataset:
        """
        Wrapper for `fillna` which ensures that additional coordinates present in
        the calling Dataset are also present in the result. The default xarray
        `fillna` implementation silently drops additional (non-indexed) coordinates
        if they need alignment (do not cover the same values in both Datasets)


        Parameters
        ----------
        ds_combine
            Dataset or DataArray to combine with the calling Dataset

        Returns
        -------
            calling Dataset calling DataArray combined with da_combine

        """

        if self._ds.pr.has_processing_info():
            raise NotImplementedError(
                "Dataset contains processing information, this is not supported yet. "
                "Use ds.pr.remove_processing_info()."
            )

        ds_start = self._ds

        coords_start = ds_start.coords
        coords_fill = ds_combine.coords
        filled = ds_start.combine_first(ds_combine)
        # check if we lost a coordinate
        missing_coords = set(coords_start).difference(set(filled.coords))
        for coord in missing_coords:
            try:
                # merge and align the coordinate
                logger.info(f"adding coord: {coord}")
                coords_merged = coords_start[coord].combine_first(coords_fill[coord])
                aligned = xr.align(filled, coords_merged, join="outer")
                # set the coordinate
                filled = aligned[0].assign_coords(
                    {coord: (coords_start[coord].dims, aligned[1].data)}
                )
            except Exception as ex:
                message = f"Could not re-add lost coordinate {coord}: {ex}"
                logger.error(message)
                raise ValueError(message) from ex

        return filled
