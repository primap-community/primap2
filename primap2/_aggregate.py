from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr

from ._accesor_base import BaseDataArrayAccessor, BaseDatasetAccessor


class DatasetAggregationAccessor(BaseDatasetAccessor):
    def fill_all_na(self, dim: Union[Sequence[str], str], value=0) -> xr.Dataset:
        """Fill NA values only where all values along the dimension(s) dim are NA.

        Example: if you have a Dataset with dimensions ``time`` and ``positions``,
        filling it along ``time`` will only fill those values where all time-points are
        NA in each DataArray.

        Parameters
        ----------
        dim: str or list of str
          Dimension(s) to evaluate. NA values are only filled if all values along these
          dimension(s) are also NA.
        value: optional
          Fill value to use. Default: 0.

        Returns
        -------
        xr.Dataset
        """
        return self._ds.map(lambda x: x.pr.fill_all_na(dim=dim, value=value))

    def sum_skip_allna(
        self,
        dim: str,
        skipna_evaluation_dims: Optional[Union[Sequence[str], str]] = None,
    ) -> xr.Dataset:
        """Sum while skipping NA values if all the values are NA.

        The sum is evaluated along the dimension ``dim`` while skipping only those NA
        values where all values along the ``skipna_evaluation_dims`` are NA.

        Example: If you have a Dataset with the dimensions ``time`` and ``position``,
        summing over ``time`` with the evaluation dimension ``position`` will skip only
        those values where _all_ values with the same ``position`` are NA in a
        DataArray.

        Parameters
        ----------
        dim: str
          Evaluation dimension to sum over.
        skipna_evaluation_dims: str or list of str, optional
          Dimension(s) to evaluate along to determine if values should be skipped.
          If omitted, all other dimensions are used.

        Returns
        -------
        xr.Dataset
        """
        if skipna_evaluation_dims is None:
            skipna_evaluation_dims = set(self._ds.dims) - {dim}

        return self.fill_all_na(dim=skipna_evaluation_dims, value=0).sum(
            dim=dim,
            skipna=False,
            keep_attrs=True,
        )


class DataArrayAggregationAccessor(BaseDataArrayAccessor):
    def fill_all_na(self, dim: Union[Sequence[str], str], value=0) -> xr.DataArray:
        """Fill NA values only where all values along the dimension(s) dim are NA.

        Example: having a data array with dimensions ``time`` and ``positions``,
        filling it along ``time`` will only fill those values where all time-points are
        NA.

        Parameters
        ----------
        dim: str or list of str
          Dimension(s) to evaluate. NA values are only filled if all values along these
          dimension(s) are also NA.
        value: optional
          Fill value to use. Default: 0.

        Returns
        -------
        xr.DataArray
        """
        return self._da.where(~np.isnan(self._da).all(dim=dim), value)

    def sum_skip_allna(
        self,
        dim: str,
        skipna_evaluation_dims: Optional[Union[Sequence[str], str]] = None,
    ) -> xr.DataArray:
        """Sum while skipping NA values if all the values are NA.

        The sum is evaluated along the dimension ``dim`` while skipping only those NA
        values where all values along the ``skipna_evaluation_dims`` are NA.

        Example: If you have a data array with the dimensions ``time`` and ``position``,
        summing over ``time`` with the evaluation dimension ``position`` will skip only
        those values where _all_ values with the same ``position`` are NA.

        Parameters
        ----------
        dim: str
          Evaluation dimension to sum over.
        skipna_evaluation_dims: str or list of str, optional
          Dimension(s) to evaluate along to determine if values should be skipped.
          If omitted, all other dimensions are used.

        Returns
        -------
        xr.DataArray
        """
        if skipna_evaluation_dims is None:
            skipna_evaluation_dims = set(self._da.dims) - {dim}

        return self.fill_all_na(dim=skipna_evaluation_dims, value=0).sum(
            dim=dim,
            skipna=False,
            keep_attrs=True,
        )
