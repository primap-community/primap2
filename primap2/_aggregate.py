import typing
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import xarray as xr

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor

DatasetOrDataArray = typing.TypeVar("DatasetOrDataArray", xr.Dataset, xr.DataArray)


def dim_names(obj: DatasetOrDataArray):
    if isinstance(obj.dims, tuple):
        return obj.dims
    else:
        return obj.dims.keys()  # type: ignore


def select_no_scalar_dimension(
    obj: DatasetOrDataArray, sel: Optional[Mapping[Hashable, Any]]
) -> DatasetOrDataArray:
    """Select but raise an error if the selection produces a new scalar dimensions.

    This can be used to guard against later broadcasting of the selection.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
    sel: selection dictionary

    Returns
    -------
    selection : same type as obj
    """
    if sel is None:
        return obj
    else:
        sele: DatasetOrDataArray = obj.loc[sel]
        if dim_names(obj) != dim_names(sele):
            raise ValueError(
                "The dimension of the selection doesn't match the dimension of the "
                "orginal dataset. Likely you used a selection casting to a scalar "
                "dimension, like sel={'axis': 'value'}. Please use "
                "sel={'axis': ['value']} instead."
            )
        return sele


class DataArrayAggregationAccessor(BaseDataArrayAccessor):
    def fill_all_na(self, dim: Union[Iterable[Hashable], str], value=0) -> xr.DataArray:
        """Fill NA values only where all values along the specified dimension(s) are NA.

        Example: having a data array with dimensions ``time`` and ``position``,
        filling it along ``time`` will only fill those values where all points that
        differ only in their ``time`` are NA, i.e. those points where all points with
        the same ``position`` are NA.

        Parameters
        ----------
        dim: str or list of str
          Dimension(s) to evaluate. NA values are only filled if all values along these
          dimension(s) are also NA.
        value: optional
          Fill value to use. Default: 0.

        Returns
        -------
        filled : xr.DataArray
        """
        if not dim:
            return self._da
        else:
            if isinstance(dim, str):
                dim = self._da.pr.dim_alias_translations.get(dim, dim)
            else:
                dim = [self._da.pr.dim_alias_translations.get(x, x) for x in dim]
            return self._da.where(~np.isnan(self._da).all(dim=dim), value)

    def sum_skip_all_na(
        self,
        dim: Hashable,
        skipna_evaluation_dims: Optional[Union[Iterable[Hashable], str]] = None,
    ) -> xr.DataArray:
        """Sum while skipping NA values if all the values are NA.

        The sum is evaluated along the dimension ``dim`` while skipping only those NA
        values where all values along the ``skipna_evaluation_dims`` are NA.

        Example: If you have a data array with the dimensions ``time`` and ``position``,
        summing over ``time`` with the evaluation dimension ``position`` will skip only
        those values where all values with the same ``position`` are NA.

        Parameters
        ----------
        dim: str
          Evaluation dimension to sum over.
        skipna_evaluation_dims: str or list of str, optional
          Dimension(s) to evaluate along to determine if values should be skipped.
          If omitted, all other dimensions are used.

        Returns
        -------
        summed : xr.DataArray
        """
        dim = self._da.pr.dim_alias_translations.get(dim, dim)

        if skipna_evaluation_dims is None:
            skipna_evaluation_dims = set(self._da.dims) - {dim}

        return self.fill_all_na(dim=skipna_evaluation_dims, value=0).sum(
            dim=dim,
            skipna=False,
            keep_attrs=True,
        )


class DatasetAggregationAccessor(BaseDatasetAccessor):
    @staticmethod
    def _apply_fill_all_na(
        da: xr.DataArray, dim: Union[Iterable[Hashable], str], value
    ) -> xr.DataArray:
        if isinstance(dim, str):
            dim = [dim]
        # dims which don't exist for a particular data variable can be excluded for that
        # data variable because if a value is NA in the other dimensions it is NA for
        # all values of a non-existing dimension
        dim = [x for x in dim if x in da.dims]
        return da.pr.fill_all_na(dim=dim, value=value)

    def fill_all_na(self, dim: Union[Iterable[Hashable], str], value=0) -> xr.Dataset:
        """Fill NA values only where all values along the specified dimension(s) are NA.

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
        filled : xr.Dataset
        """
        if isinstance(dim, str):
            dim = self._ds.pr.dim_alias_translations.get(dim, dim)
        else:
            dim = [self._ds.pr.dim_alias_translations.get(x, x) for x in dim]
        return self._ds.map(
            self._apply_fill_all_na, dim=dim, value=value, keep_attrs=True
        )

    def sum_skip_all_na(
        self,
        dim: Hashable,
        skipna_evaluation_dims: Optional[Union[Iterable[Hashable], str]] = None,
    ) -> xr.Dataset:
        """Sum while skipping NA values if all the values are NA.

        The sum is evaluated along the dimension ``dim`` while skipping only those NA
        values where all values along the ``skipna_evaluation_dims`` are NA.

        Example: If you have a Dataset with the dimensions ``time`` and ``position``,
        summing over ``time`` with the evaluation dimension ``position`` will skip only
        those values where all values with the same ``position`` are NA in a
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
        summed : xr.Dataset
        """
        dim = self._ds.pr.dim_alias_translations.get(dim, dim)

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
        skipna_evaluation_dims: Sequence[str] = tuple(),
    ) -> xr.DataArray:
        """The sum of gas basket contents converted using the global warming potential
        of the gas basket.

        Parameters
        ----------
        basket: str
          The name of the gas basket. A value from `ds.keys()`.
        basket_contents: list of str
          The name of the gases in the gas basket. The sum of all basket_contents
          equals the basket. Values from `ds.keys()`.
        skipna_evaluation_dims: list of str, optional
          Dimensions which should be evaluated to determine if NA values should be
          skipped entirely if missing fully. By default, no NA values are skipped.

        Returns
        -------
        summed : xr.DataArray
        """

        basket_contents_converted = xr.Dataset()
        basket_da = self._ds[basket]
        for var in basket_contents:
            da: xr.DataArray = self._ds[var]
            basket_contents_converted[var] = da.pr.convert_to_gwp_like(like=basket_da)

        basket_contents_converted_da: xr.DataArray = basket_contents_converted.to_array(
            "entity"
        )

        da = basket_contents_converted_da.pr.sum_skip_all_na(
            dim="entity",
            skipna_evaluation_dims=skipna_evaluation_dims,
        )
        da.attrs["gwp_context"] = basket_da.attrs["gwp_context"]
        da.attrs["entity"] = basket_da.attrs["entity"]
        da.name = basket_da.name
        return da

    def fill_na_gas_basket_from_contents(
        self,
        *,
        basket: str,
        basket_contents: Sequence[str],
        sel: Optional[Mapping[Hashable, Sequence]] = None,
        skipna_evaluation_dims: Sequence[str] = tuple(),
    ) -> xr.DataArray:
        """Fill NA values in a gas basket using the sum of its contents.

        To calculate the sum of the gas basket contents, the global warming potential
        context defined on the gas basket will be used.

        Parameters
        ----------
        basket: str
          The name of the gas basket. A value from `ds.keys()`.
        basket_contents: list of str
          The name of the gases in the gas basket. The sum of all basket_contents
          equals the basket. Values from `ds.keys()`.
        sel: Selection dict, optional
          If the filling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          filling will be done on `ds.loc[sel]`.
        skipna_evaluation_dims: list of str, optional
          Dimensions which should be evaluated to determine if NA values should be
          skipped entirely if missing fully. By default, no NA values are skipped.

        Returns
        -------
        filled : xr.DataArray
        """
        ds_sel = select_no_scalar_dimension(self._ds, sel)
        return self._ds[basket].fillna(
            ds_sel.pr.gas_basket_contents_sum(
                basket=basket,
                basket_contents=basket_contents,
                skipna_evaluation_dims=skipna_evaluation_dims,
            )
        )
