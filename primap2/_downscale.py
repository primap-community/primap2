from typing import Dict, Hashable, Optional, Sequence

import xarray as xr

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor
from ._aggregate import select_no_scalar_dimension
from ._units import ureg


class DataArrayDownscalingAccessor(BaseDataArrayAccessor):
    def downscale_timeseries(
        self,
        *,
        dim: Hashable,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        check_consistency: bool = True,
        sel: Optional[Dict[Hashable, Sequence]] = None,
        skipna_evaluation_dims: Sequence[Hashable] = tuple(),
    ) -> xr.DataArray:
        """Downscale timeseries along a dimension using a basket defined on a
        broader timeseries.

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

        Returns
        -------
        downscaled: xr.DataArray
        """
        da_sel = select_no_scalar_dimension(self._da, sel)

        basket_contents_da = da_sel.loc[{dim: basket_contents}]
        basket_da = da_sel.loc[{dim: basket}]

        basket_sum = basket_contents_da.pr.sum_skip_all_na(
            dim=dim, skipna_evaluation_dims=skipna_evaluation_dims
        )

        if check_consistency:
            deviation: xr.DataArray = abs(basket_da / basket_sum - 1)
            devmax = deviation.max()
            if devmax > 0.01:
                raise ValueError(
                    f"Sum of the basket_contents {basket_contents!r} deviates"
                    f" {devmax * 100} % from the basket"
                    f" {basket!r}, which is more than the allowed 1 %. "
                    "To continue regardless, set check_consistency=False."
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

        downscaled: xr.DataArray = basket_da * shares

        return self._da.fillna(downscaled)


class DatasetDownscalingAccessor(BaseDatasetAccessor):
    def downscale_timeseries(
        self,
        *,
        dim: Hashable,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        check_consistency: bool = True,
        sel: Optional[Dict[Hashable, Sequence]] = None,
        skipna_evaluation_dims: Sequence[Hashable] = tuple(),
    ) -> xr.Dataset:
        """Downscale timeseries along a dimension using a basket defined on a
        broader timeseries.

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

        Notes
        -----
        To downscale along the entity dimension, i.e. to downscale gases, see
        :py:meth:`downscale_gas_timeseries`, which handles gwp conversions
        appropriately.

        Returns
        -------
        downscaled: xr.Dataset
        """
        ds_sel = select_no_scalar_dimension(self._ds, sel)

        basket_contents_ds = ds_sel.loc[{dim: basket_contents}]
        basket_ds = ds_sel.loc[{dim: basket}]

        basket_sum = basket_contents_ds.pr.sum_skip_all_na(
            dim=dim, skipna_evaluation_dims=skipna_evaluation_dims
        )

        if check_consistency:
            deviation = abs(basket_ds / basket_sum - 1)
            devmax = deviation.to_array().max()
            if devmax > 0.01:
                raise ValueError(
                    f"Sum of the basket_contents {basket_contents!r} deviates"
                    f" {devmax * 100} % from the basket"
                    f" {basket!r}, which is more than the allowed 1 %. "
                    "To continue regardless, set check_consistency=False."
                )

        # inter- and extrapolate
        shares: xr.Dataset = (
            (basket_contents_ds / basket_sum)
            .pint.to({x: "" for x in basket_contents_ds.keys()})
            .pint.dequantify()
            .interpolate_na(dim="time", method="linear")
            .ffill(dim="time")
            .bfill(dim="time")
        )

        downscaled: xr.Dataset = basket_ds * shares

        return self._ds.fillna(downscaled)

    def downscale_gas_timeseries(
        self,
        *,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        check_consistency: bool = True,
        sel: Optional[Dict[Hashable, Sequence]] = None,
        skipna_evaluation_dims: Sequence[Hashable] = tuple(),
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

        Returns
        -------
        downscaled: xr.Dataset
        """
        ds_sel = select_no_scalar_dimension(self._ds, sel)

        basket_contents_converted = xr.Dataset()
        da_basket = ds_sel[basket]
        for var in basket_contents:
            da: xr.DataArray = ds_sel[var]
            basket_contents_converted[var] = da.pr.convert_to_gwp_like(like=da_basket)

        da_basket_contents: xr.DataArray = basket_contents_converted.to_array("entity")
        basket_sum = da_basket_contents.pr.sum_skip_all_na(
            dim="entity", skipna_evaluation_dims=skipna_evaluation_dims
        )

        if check_consistency:
            deviation = abs(da_basket / basket_sum - 1)
            devmax = deviation.max().item()
            if devmax > 0.01:
                raise ValueError(
                    f"Sum of the basket_contents {basket_contents!r} deviates"
                    f" {devmax * 100} % from the basket"
                    f" {basket!r}, which is more than the allowed 1 %. "
                    "To continue regardless, set check_consistency=False."
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

        downscaled = da_basket * shares

        downscaled_converted = xr.Dataset()

        with ureg.context(da_basket.attrs["gwp_context"]):
            for var in basket_contents:
                downscaled_converted[var] = ds_sel[var].fillna(
                    downscaled[var].pint.to(ds_sel[var].pint.units)
                )

        return self._ds.fillna(downscaled_converted)
