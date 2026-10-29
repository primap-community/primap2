"""workaround for xarray coordinate bug

Additional coordinates are lost during operations that require aligning if the
coordinates differ for the structures to be aligned.

see https://github.com/pydata/xarray/issues/9124

"""

import pint
import xarray as xr
from loguru import logger

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor
from ._processing_info import records_processing_step


def _readd_lost_coords(start, other, method: str):
    """Call ``start.<method>(other)`` and re-add coordinates of ``start`` lost on the way."""
    coords_start = start.coords
    coords_other = other.coords
    filled = getattr(start, method)(other)
    # check if we lost a coordinate
    missing_coords = set(coords_start).difference(set(filled.coords))
    for coord in missing_coords:
        try:
            # merge and align the coordinate
            logger.info(f"adding coord: {coord}")
            coords_merged = getattr(coords_start[coord], method)(coords_other[coord])
            aligned = xr.align(filled, coords_merged, join="outer")
            # set the coordinate
            filled = aligned[0].assign_coords({coord: (coords_start[coord].dims, aligned[1].data)})
        except Exception as ex:
            message = f"Could not re-add lost coordinate {coord}: {ex}"
            logger.error(message)
            raise ValueError(message) from ex

    return filled


class DataArrayFillAccessor(BaseDataArrayAccessor):
    def fillna(self: xr.DataArray, da_fill: xr.DataArray) -> xr.DataArray:
        """Fill missing information from other array.

        Wrapper for :py:meth:`xarray.DataArray.fillna` which ensures that additional
        coordinates present in the calling DataArray are also present in the result.
        The default :py:meth:`xarray.DataArray.fillna` implementation silently drops
        additional (non-indexed) coordinates if they need alignment (do not cover the
        same values in both DataArrays)

        Parameters
        ----------
        da_fill
            DataArray used to fill nan values in the calling DataArray

        Returns
        -------
            filled
                calling DataArray where nan values are filled from da_fill where possible
        """
        return _readd_lost_coords(self._da, da_fill, "fillna")

    def combine_first(self: xr.DataArray, da_combine: xr.DataArray) -> xr.DataArray:
        """Combine data from multiple arrays.

        Wrapper for :py:meth:`xarray.DataArray.combine_first` which ensures that
        additional coordinates present in the calling dataset are also present in the
        result. The default :py:meth:`xarray.DataArray.combine_first`
        implementation silently drops additional (non-indexed) coordinates if they need
        alignment (do not cover the same values in both DataArrays)

        Parameters
        ----------
        da_combine
            DataArray used to combine with the calling DataArray

        Returns
        -------
            combined
                calling DataArray combined with da_combine
        """
        return _readd_lost_coords(self._da, da_combine, "combine_first")


class DatasetFillAccessor(BaseDatasetAccessor):
    def fillna(
        self,
        ds_fill: xr.Dataset | xr.DataArray | float | pint.Quantity,
    ) -> xr.Dataset:
        """Fill missing information from other dataset.

        Wrapper for :py:meth:`xarray.Dataset.fillna` which ensures that additional
        coordinates present in the calling Dataset are also present in the result.
        The default :py:meth:`xarray.Dataset.fillna` implementation silently drops
        additional (non-indexed) coordinates if they need alignment (do not cover the
        same values in both Datasets)

        Parameters
        ----------
        ds_fill
            Dataset, DataArray or single value used to fill nan values in the calling
            dataset. If the calling dataset has processing information, a Dataset needs
            processing information as well, see
            :py:meth:`xarray.DataArray.pr.with_creation_step`, and a single value is recorded
            as the creation of the filled values.

        Returns
        -------
            filled
                calling Dataset where nan values are filled from ds_fill where possible
        """
        if isinstance(ds_fill, xr.Dataset | xr.DataArray):
            return self._fillna(ds_fill)
        return self._fillna_value(ds_fill)

    @records_processing_step(
        "self",
        function="fillna",
        description="filled missing values with {value}",
        source="constant({value})",
    )
    def _fillna_value(self, value: float | pint.Quantity) -> xr.Dataset:
        return self._ds.fillna(value)

    @records_processing_step(
        "self",
        other="ds_fill",
        function="fillna",
        description="filled missing values",
    )
    def _fillna(self, ds_fill: xr.Dataset | xr.DataArray) -> xr.Dataset:
        return _readd_lost_coords(self._ds, ds_fill, "fillna")

    @records_processing_step(
        "self",
        other="ds_combine",
        function="combine_first",
        description="combined with other data",
    )
    def combine_first(
        self,
        ds_combine: xr.Dataset | xr.DataArray,
    ) -> xr.Dataset:
        """Combine with data from other dataset or array.

        Wrapper for :py:meth:`xarray.Dataset.combine_first` which ensures that additional
        coordinates present in the calling Dataset are also present in the result.
        The default :py:meth:`xarray.Dataset.combine_first` implementation silently
        drops additional (non-indexed) coordinates
        if they need alignment (do not cover the same values in both Datasets)

        Parameters
        ----------
        ds_combine
            Dataset or DataArray to combine with the calling Dataset

        Returns
        -------
            combined
                calling Dataset calling DataArray combined with da_combine
        """
        return _readd_lost_coords(self._ds, ds_combine, "combine_first")
