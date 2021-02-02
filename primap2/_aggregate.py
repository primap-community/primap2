from typing import Dict, Optional, Sequence, Union

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

    def gas_basket_contents_sum(
        self,
        *,
        basket: str,
        basket_contents: Sequence[str],
        sel: Optional[Dict] = None,
    ) -> xr.DataArray:
        """The sum of gas basket contents converted using the global warming potential
        of the gas basket.

        Parameters
        ----------
        basket: str
          The name of the gas basket for which values are known at higher temporal
          resolution and/or for a wider range. A value from `ds.keys()`.
        basket_contents: list of str
          The name of the gases in the gas basket. The sum of all basket_contents
          equals the basket. Values from `ds.keys()`.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on `ds.loc[sel]`.

        Returns
        -------
        summed : xr.DataArray
        """
        if sel is not None:
            ds_sel = self._ds.loc[sel]
        else:
            ds_sel = self._ds

        basket_contents_converted = xr.Dataset()
        basket_da = ds_sel[basket]
        for var in basket_contents:
            da: xr.DataArray = ds_sel[var]
            basket_contents_converted[var] = da.pr.convert_to_gwp_like(like=basket_da)

        basket_contents_converted_da: xr.DataArray = basket_contents_converted.to_array(
            "entity"
        )

        da = basket_contents_converted_da.pr.sum_skip_allna(
            dim="entity",
            skipna_evaluation_dims=["date"],
        )
        da.attrs["gwp_context"] = basket_da["gwp_context"]
        da.attrs["entity"] = basket_da["entity"]
        return da

    def fill_na_gas_basket_from_contents(
        self,
        *,
        basket: str,
        basket_contents: Sequence[str],
        sel: Optional[Dict] = None,
    ) -> xr.DataArray:
        """Fill NA values in a gas basket using the sum of its contents.

        To calculate the sum of the gas basket contents, the global warming potential
        context defined on the gas basket will be used.

        Parameters
        ----------
        basket: str
          The name of the gas basket for which values are known at higher temporal
          resolution and/or for a wider range. A value from `ds.keys()`.
        basket_contents: list of str
          The name of the gases in the gas basket. The sum of all basket_contents
          equals the basket. Values from `ds.keys()`.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on `ds.loc[sel]`.

        Returns
        -------
        filled : xr.DataArray
        """
        return self._ds[basket].fillna(
            self.gas_basket_contents_sum(
                basket=basket, basket_contents=basket_contents, sel=sel
            )
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
