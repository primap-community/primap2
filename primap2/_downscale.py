from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor
from ._aggregate import select_no_scalar_dimension
from ._units import ureg

# Needed for downscaling operations
xr.set_options(use_numbagg=True)


class DataArrayDownscalingAccessor(BaseDataArrayAccessor):
    def downscale_timeseries(
        self,
        *,
        dim: Hashable,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        check_consistency: bool = True,
        sel: dict[Hashable, Sequence] | None = None,
        skipna_evaluation_dims: None | Sequence[Hashable] = None,
        skipna: bool = True,
        tolerance: float = 0.01,
    ) -> xr.DataArray:
        """Downscale timeseries along a dimension using a basket defined on a broader timeseries.

        This is useful if you have data for many points in time for a total, for example
        the entire Energy sector, and higher-resolution data (e.g. fossil and non-fossil
        energies separately) for only a few points in time. In the example, the Energy
        sector is the ``basket`` and fossil and non-fossil energies are the basket
        contents.

        From any time points where all the basket contents are known, the
        relative shares of the basket contents are determined, and then interpolated
        linearly and extrapolated constantly to the full timeseries. The shares are then
        used to downscale the basket to its contents, which is used to fill gaps in the
        timeseries of the basket contents.

        If the data to downscale contains only 0 and NaN the result will be all
        zero timeseries. If the data is only for individual basket and basket_content
        values, the resulting downscaled data points are zero but the shares for other years
        are not influenced

        Parameters
        ----------
        dim: str
          The name of the dimension which contains the basket and its contents, has to
          be one of the dimensions in ``da.dims``.
        basket: str
          The name of the super-category for which values are known at higher temporal
          resolution and/or for a wider range. A value from ``da[dimension]``.
        basket_contents: list of str
          The name of the sub-categories. The sum of all sub-categories equals the
          basket. Values from ``da[dimension]``.
        check_consistency: bool, default True
          If for all points where the basket and all basket_contents are defined,
          it should be checked if the sum of the basket_contents actually equals
          the basket. A ``ValueError`` is raised if the consistency check fails.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on ``da.loc[sel]``.
        skipna_evaluation_dims: list of str, optional
          Dimensions which should be evaluated to determine if NA values should be
          skipped entirely if missing fully. By default, no NA values are skipped.
        skipna: bool, default True
          If true it will be passed on to xarray's ds.sum function with min_count=1
          for the calculation of the basket.
          The effect is that NA values in a sum will be ignored and treated as zero
          in the sum unless all values are NA which results in NA.
        tolerance: float
          If given it overrides the default tolerance for deviations of sums of
          individual timeseries to given aggregate timeseries. Default is 0.01 (1%)

        Returns
        -------
        downscaled: xr.DataArray
        """
        da_sel = select_no_scalar_dimension(self._da, sel)

        basket_contents_da = da_sel.loc[{dim: basket_contents}]
        basket_da = da_sel.loc[{dim: basket}]

        if skipna_evaluation_dims is not None:
            if skipna:
                raise ValueError(
                    "Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied, not both."
                )
            else:
                skipna = None

        basket_sum = basket_contents_da.pr.sum(
            dim=dim,
            skipna=skipna,
            min_count=1,
            skipna_evaluation_dims=skipna_evaluation_dims,
        )

        if check_consistency:
            deviation: xr.DataArray = abs(basket_da / basket_sum - 1)
            devmax = float(deviation.max().pint.dequantify().data)
            if devmax > tolerance:
                raise ValueError(
                    f"Sum of the basket_contents {basket_contents!r} deviates"
                    f" {devmax * 100} % from the basket"
                    f" {basket!r}, which is more than the allowed {tolerance * 100}%. "
                    "To continue regardless, set check_consistency=False."
                )

        # treat zeros
        basket_both_zero = basket_da.where((basket_da == 0) & (basket_sum == 0))
        basket_sum_zero = basket_da.where((basket_da != 0) & (basket_sum == 0))
        if not basket_sum_zero.isnull().all():
            # this can only happen if check consistency = False,
            # so we could also remove it and say whoever switches it of has
            # to deal with the consequences
            error_message = generate_error_message(basket_sum_zero)
            raise ValueError(f"pr.downscale_timeseries {error_message}")

        any_nonzero = basket_sum.where(basket_sum != 0).notnull().any()

        # treat the case where all data is zero or NaN
        if (not basket_both_zero.isnull().all()) & (not any_nonzero):
            # create fake basket_contents_da and basket_sum which contain
            # 1 for all data points where NaNs are either in the sum or the basket
            # this will later lead to equal shares for the NaNs.
            units = basket_contents_da.pint.units
            basket_da_squeeze = basket_da.drop(dim)
            basket_contents_da = basket_contents_da.where(
                (basket_da_squeeze != 0) | (basket_sum != 0), 1 * units
            )
            basket_sum = basket_contents_da.pr.sum(
                dim=dim,
                skipna=skipna,
                min_count=1,
                skipna_evaluation_dims=skipna_evaluation_dims,
            )

        # inter- and extrapolate
        shares: xr.DataArray = (
            (basket_contents_da / basket_sum)
            .pint.to("")
            .pint.dequantify()
            .interpolate_na(dim="time", method="linear")
            .ffill(dim="time")
            .bfill(dim="time")
        )

        # treat the case where there are zero values in basket_sum and basket_da but also
        # non-zero data
        if (not basket_both_zero.isnull().all()) & any_nonzero:
            shares = shares.where((basket_da != 0) | (basket_sum != 0), 0)

        downscaled: xr.DataArray = basket_da * shares

        return self._da.fillna(downscaled)

    def downscale_timeseries_by_shares(
        self,
        *,
        dim: Hashable,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        basket_contents_shares: xr.DataArray,
    ) -> xr.DataArray:
        """Downscale timeseries along a dimension using defined shares for each timestep.

        This is useful if you have data points for a total, and you don't have any
        data for the higher resolution, but you do have the shares for the higher
        resolution from another source. For example, you have the total energy, and
        you know the shares of the sub-sectors 1.A and 1.B for each year from another
        source.

        Parameters
        ----------
        dim : Hashable
            The dimension along which to perform the downscaling (e.g., "category" or "area").
        basket : Hashable
            The label of the aggregate group (e.g., "1.A" for a category) whose value will be
            redistributed.
        basket_contents : Sequence of Hashable
            The labels of the subgroups (e.g., ["1.A.1", "1.A.2", "1.A.3"])
            that make up the `basket`.
        basket_contents_shares : xr.DataArray
            The shares to use for downscaling.

        Returns
        -------
        xr.DataArray
            A new datarray with variables downscaled along `dim` using the provided shares.
        """

        basket_contents_shares = basket_contents_shares.pr.loc[{dim: basket_contents}]

        # normalise shares
        basket_contents_shares = basket_contents_shares / basket_contents_shares.sum(dim=dim)

        # Make sure the result won't be empty.
        # xarray will try to match every indexed coordinate (coordinates with *)
        # if they don't match the result will be empty
        array_to_downscale = self._da.pr.loc[{dim: basket}]

        # aligned is a tuple of datarray with aligned coordinates
        aligned = xr.align(array_to_downscale, basket_contents_shares, join="inner")
        if all([i.size == 0 for i in aligned]):
            raise ValueError(
                "No overlap found between the input data and the provided shares."
                "Check coordinate alignment"
            )

        downscaled = array_to_downscale * basket_contents_shares

        return self._da.pr.set(dim=dim, key=basket_contents, value=downscaled)


class DatasetDownscalingAccessor(BaseDatasetAccessor):
    def downscale_timeseries(
        self,
        *,
        dim: Hashable,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        check_consistency: bool = True,
        sel: dict[Hashable, Sequence] | None = None,
        skipna_evaluation_dims: Sequence[Hashable] | None = None,
        skipna: bool = True,
        tolerance: float = 0.01,
    ) -> xr.Dataset:
        """Downscale timeseries along a dimension using a basket defined on a broader timeseries.

        This is useful if you have data for many points in time for a total, for example
        the entire Energy sector, and higher-resolution data (e.g. fossil and non-fossil
        energies separately) for only a few points in time. In the example, the Energy
        sector is the ``basket`` and fossil and non-fossil energies are the basket
        contents.

        From any time points where all the basket contents are known, the
        relative shares of the basket contents are determined, and then interpolated
        linearly and extrapolated constantly to the full timeseries. The shares are then
        used to downscale the basket to its contents, which is used to fill gaps in the
        timeseries of the basket contents.

        If the data to downscale contains only 0 an NaN the result will be all
        zero timeseries. If the data is only for individual basket and basket_content
        values, the resulting downscaled data points are zero but the shares for other years
        are not influenced

        Parameters
        ----------
        dim: str
          The name of the dimension which contains the basket and its contents, has to
          be one of the dimensions in ``ds.dims``.
        basket: str
          The name of the super-category for which values are known at higher temporal
          resolution and/or for a wider range. A value from ``ds[dimension]``.
        basket_contents: list of str
          The name of the sub-categories. The sum of all sub-categories equals the
          basket. Values from ``ds[dimension]``.
        check_consistency: bool, default True
          If for all points where the basket and all basket_contents are defined,
          it should be checked if the sum of the basket_contents actually equals
          the basket. A ``ValueError`` is raised if the consistency check fails.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on ``ds.loc[sel]``.
        skipna_evaluation_dims: list of str, optional
          Dimensions which should be evaluated to determine if NA values should be
          skipped entirely if missing fully. By default, no NA values are skipped.
        skipna: bool, default True
          If true it will be passed on to xarray's ds.sum function with min_count=1 for
          the calculation of the basket.
          The effect is that NA values in a sum will be ignored and treated as zero
          in the sum unless all values are NA which results in NA.
        tolerance: float
          If given it overrides the default tolerance for deviations of sums of
          individual timeseries to given aggregate timeseries. Default is 0.01 (1%)

        Notes
        -----
        To downscale along the entity dimension, i.e. to downscale gases, see
        :py:meth:`downscale_gas_timeseries`, which handles gwp conversions
        appropriately.

        Returns
        -------
        downscaled: xr.Dataset
        """
        if self._ds.pr.has_processing_info():
            raise NotImplementedError(
                "Dataset contains processing information, this is not supported yet. "
                "Use ds.pr.remove_processing_info()."
            )

        downscaled = self._ds.copy()
        for var in self._ds.data_vars:
            downscaled[var] = downscaled[var].pr.downscale_timeseries(
                dim=dim,
                basket=basket,
                basket_contents=basket_contents,
                check_consistency=check_consistency,
                sel=sel,
                skipna_evaluation_dims=skipna_evaluation_dims,
                skipna=skipna,
                tolerance=tolerance,
            )

        return downscaled

    def downscale_gas_timeseries(
        self,
        *,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        check_consistency: bool = True,
        sel: dict[Hashable, Sequence] | None = None,
        skipna_evaluation_dims: Sequence[Hashable] | None = None,
        skipna: bool = True,
        tolerance: float = 0.01,
    ) -> xr.Dataset:
        """Downscale a gas basket defined on a broader timeseries to its contents
        known on fewer time points.

        This is useful if you have data for many points in time for a gas basket, for
        example KYOTOGHG, and higher-resolution data (e.g. the individual green house
        gases included in KYOTOGHG) for only a few points in time.
        From any time points where all the basket contents are known, the relative
        shares of the basket contents are determined, and then interpolated linearly
        and extrapolated constantly to the full timeseries. The shares are then
        used to downscale the basket to its contents, which is used to fill gaps in the
        timeseries of the basket contents.
        Basket contents are converted to the units of the basket for downscaling, and
        converted back afterwards; for both conversions the gwp conversions of the
        basket are used.

        If the data to downscale contains only 0 an NaN the result will be all
        zero timeseries. If the data is only for individual basket and basket_content
        values, the resulting downscaled data points are zero but the shares for other years
        are not influenced

        Parameters
        ----------
        basket: str
          The name of the gas basket for which values are known at higher temporal
          resolution and/or for a wider range. A value from ``ds.keys()``.
        basket_contents: list of str
          The name of the gases in the gas basket. The sum of all basket_contents
          equals the basket. Values from ``ds.keys()``.
        check_consistency: bool, default True
          If for all points where the basket and all basket_contents are defined,
          it should be checked if the sum of the basket_contents actually equals
          the basket. A ``ValueError`` is raised if the consistency check fails.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on ``ds.loc[sel]``.
        skipna_evaluation_dims: list of str, optional
          Dimensions which should be evaluated to determine if NA values should be
          skipped entirely if missing fully. By default, no NA values are skipped.
        skipna: bool, optional
          If true it will be passed on to xarray's ds.sum function with min_count=1 for
          the calculation of the basket.
          The effect is that NA values in a sum will be ignored and treated as zero
          in the sum unless all values are NA which results in NA.
        tolerance: float
          If given it overrides the default tolerance for deviations of sums of
          individual timeseries to given aggregate timeseries. Default is 0.01 (1%)

        Returns
        -------
        downscaled: xr.Dataset
        """
        if self._ds.pr.has_processing_info():
            raise NotImplementedError(
                "Dataset contains processing information, this is not supported yet. "
                "Use ds.pr.remove_processing_info()."
            )

        ds_sel = select_no_scalar_dimension(self._ds, sel)

        basket_contents_converted = xr.Dataset()
        basket_da = ds_sel[basket]
        for var in basket_contents:
            da: xr.DataArray = ds_sel[var]
            basket_contents_converted[var] = da.pr.convert_to_gwp_like(like=basket_da)

        if skipna_evaluation_dims is not None:
            if skipna:
                raise ValueError(
                    "Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied, not both."
                )
            else:
                skipna = None

        basket_sum = basket_contents_converted.pr.sum(
            dim="entity",
            skipna=skipna,
            min_count=1,
            skipna_evaluation_dims=skipna_evaluation_dims,
        )

        if check_consistency:
            deviation = abs(basket_da / basket_sum - 1)
            devmax = deviation.max().item()
            if devmax > tolerance:
                raise ValueError(
                    f"Sum of the basket_contents {basket_contents!r} deviates"
                    f" {devmax * 100} % from the basket"
                    f" {basket!r}, which is more than the allowed "
                    f"{tolerance * 100}%. "
                    f" {deviation}"
                    "To continue regardless, set check_consistency=False."
                )

        # treat zeros
        basket_both_zero = basket_da.where((basket_da == 0) & (basket_sum == 0))
        basket_sum_zero = basket_da.where((basket_da != 0) & (basket_sum == 0))
        if not basket_sum_zero.isnull().all():
            # this can only happen if check consistency = False,
            # so we could also remove it and say whoever switches it of has
            # to deal with the consequences
            error_message = generate_error_message(basket_sum_zero)
            raise ValueError(f"pr.downscale_gas_timeseries {error_message}")

        any_nonzero = basket_sum.where(basket_sum != 0).notnull().any()

        # treat the case where all data is zero or NaN
        if (not basket_both_zero.isnull().all()) & (not any_nonzero):
            unit_content = str(basket_da.pint.units)
            basket_contents_converted = basket_contents_converted.where(
                (basket_da != 0) | (basket_sum != 0), 1 * ureg(unit_content)
            )
            basket_sum = basket_contents_converted.pr.sum(
                dim="entity",
                skipna=skipna,
                min_count=1,
                skipna_evaluation_dims=skipna_evaluation_dims,
            )

        # inter- and extrapolate
        shares = (
            (basket_contents_converted / basket_sum)
            .pint.to({x: "" for x in basket_contents_converted.keys()})
            .pint.dequantify()
            .interpolate_na(dim="time", method="linear")
            .ffill(dim="time")
            .bfill(dim="time")
        )

        # treat the case where there are zero values in basket_sum and basket_da but also
        # non-zero data
        if (not basket_both_zero.isnull().all()) & any_nonzero:
            shares = shares.where((basket_da != 0) | (basket_sum != 0), 0)

        downscaled = basket_da * shares

        downscaled_converted = xr.Dataset()

        with ureg.context(basket_da.attrs["gwp_context"]):
            for var in basket_contents:
                downscaled_converted[var] = ds_sel[var].fillna(
                    downscaled[var].pint.to(ds_sel[var].pint.units)
                )

        return self._ds.pr.fillna(downscaled_converted)

    def downscale_timeseries_by_shares(
        self,
        *,
        dim: Hashable,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        basket_contents_shares: xr.DataArray | xr.Dataset,
    ) -> xr.Dataset:
        """Downscale timeseries along a dimension using defined shares for each timestep.

        This is useful if you have data points for a total, and you don't have any
        data for the higher resolution, but you do have the shares for the higher
        resolution from another source. For example, you have the total energy, and
        you know the shares of the sub-sectors 1.A and 1.B for each year from another
        source.

        Parameters
        ----------
        dim : Hashable
            The dimension along which to perform the downscaling (e.g., "category" or "area").
        basket : Hashable
            The label of the aggregate group (e.g., "1.A" for a category) whose value will be
            redistributed.
        basket_contents : Sequence of Hashable
            The labels of the subgroups (e.g., ["1.A.1", "1.A.2", "1.A.3"])
            that make up the `basket`.
        basket_contents_shares : xr.DataArray or xr.Dataset
            The shares to use for downscaling.

        Returns
        -------
        xr.Dataset
            A new dataset with variables downscaled along `dim` using the provided shares.
        """
        ds = self._ds.copy()
        downscaled_dict = {}
        for var in self._ds.data_vars:
            if isinstance(basket_contents_shares, xr.Dataset):
                # if the reference does not specify shares for this variable, skip it
                if var not in basket_contents_shares.data_vars:
                    logger.warning(f"{var} is not in reference data. Skipping it")
                    continue
                basket_contents_shares_arr = basket_contents_shares[var]
            else:
                basket_contents_shares_arr = basket_contents_shares
            downscaled_dict[var] = ds[var].pr.downscale_timeseries_by_shares(
                dim=dim,
                basket=basket,
                basket_contents=basket_contents,
                basket_contents_shares=basket_contents_shares_arr,
            )

        return xr.Dataset(downscaled_dict).assign_attrs(ds.attrs)


def generate_error_message(da_error: xr.DataArray) -> str:
    """Generate error message for zero sum data.

    based on generate_log_message for pr.merge
    Strategy:

    * remove all length-1 dimensions and put them in description
    * convert the rest into a pandas dataframe for nice printing
    """
    scalar_dims = [dim for dim in da_error.dims if len(da_error[dim]) == 1]
    scalar_dims_format = []
    for dim in scalar_dims:
        if pd.api.types.is_datetime64_any_dtype(da_error[dim]):
            ts = pd.Timestamp(da_error[dim][0].values)
            # optimization for the common case where we have data on a yearly or
            # more coarse basis
            if ts == pd.Timestamp(year=ts.year, month=1, day=1):
                scalar_dims_format.append(f"{dim}={ts.strftime('%Y')}")
            else:
                scalar_dims_format.append(f"{dim}={ts!s}")
        else:
            scalar_dims_format.append(f"{dim}={da_error[dim].item()}")
    scalar_dims_str = ", ".join(scalar_dims_format)
    da_error_dequ = da_error.squeeze(drop=True).pint.dequantify()
    if np.ndim(da_error_dequ.data) == 0:
        errors_str = ""
    else:
        errors_str = da_error_dequ.to_dataframe().dropna().to_string()

    return (
        "error: found zero basket content sum for non-zero basket data"
        f" for {scalar_dims_str}:\n"
        f"({da_error.name})\n" + errors_str
    )
