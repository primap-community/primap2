from typing import Dict, Optional, Sequence

import xarray as xr

from ._accesor_base import BaseDataArrayAccessor, BaseDatasetAccessor
from ._units import ureg


class DatasetDownscalingAccessor(BaseDatasetAccessor):
    def downscale_timeseries(
        self,
        *,
        dimension: str,
        basket: str,
        basket_contents: Sequence[str],
        check_consistency: bool = True,
        sel: Optional[Dict] = None,
        skip_all_na: bool = False,
    ) -> xr.Dataset:
        """Downscale timeseries along a dimension using a basket defined on a
        broader timeseries.

        This is useful if you have data for many points in time for a total, for example
        the entire Energy sector, and higher-resolution data (e.g. fossil and non-fossil
        energies separately) for only a few points in time. In the example, the Energy
        sector is the `basket` and fossil and non-fossil energies are the basket
        contents.
        From any time points where all the basket contents are known, the
        relative shares of the basket contents are determined, and then interpolated
        linearly and extrapolated constantly to the full timeseries. The shares are then
        used to downscale the basket to its contents, which is used to fill gaps in the
        timeseries of the basket contents.

        Parameters
        ----------
        dimension: str
          The name of the dimension which contains the basket and its contents, has to
          be one of the dimensions in `ds.dims`.
        basket: str
          The name of the super-category for which values are known at higher temporal
          resolution and/or for a wider range. A value from `ds[dimension]`.
        basket_contents: list of str
          The name of the sub-categories. The sum of all sub-categories equals the
          basket. Values from `ds[dimension]`.
        check_consistency: bool, default True
          If for all points where the basket and all basket_contents are defined,
          it should be checked if the sum of the basket_contents actually equals
          the basket. A ValueError is raised if the consistency check fails.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on ds.loc[sel].
        skip_all_na: bool, default False
          If basket_contents which are NA for all points in a time series are skipped
          entirely.

        Notes
        -----
        To downscale along the entity dimension, i.e. to downscale gases, see
        downscale_gas_timeseries, which handles gwp conversions appropriately.

        Returns
        -------
        downscaled: xr.Dataset
        """
        if sel is not None:
            ds_sel = self._ds.loc[sel]
        else:
            ds_sel = self._ds

        basket_contents_ds = ds_sel.loc[{dimension: basket_contents}]
        basket_ds = ds_sel.loc[{dimension: basket}]

        if skip_all_na:
            basket_sum = basket_contents_ds.pr.sum_skip_all_na(
                dim=dimension,
                skipna_evaluation_dims="date",
            )
        else:
            basket_sum = basket_contents_ds.sum(
                dimension, skipna=False, keep_attrs=True
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
            .interpolate_na(dim="date", method="linear")
            .ffill(dim="date")
            .bfill(dim="date")
        )

        downscaled: xr.Dataset = basket_ds * shares

        return self._ds.fillna(downscaled)

    def downscale_gas_timeseries(
        self,
        *,
        basket: str,
        basket_contents: Sequence[str],
        check_consistency: bool = True,
        sel: Optional[Dict] = None,
        skip_all_na: bool = False,
    ) -> xr.Dataset:
        """Downscale a gas basket defined on a broader timeseries to its contents
        known for fewer time points.

        This is useful if you have data for many points in time for a gas basket, for
        example KYOTOGHG, and higher-resolution data (e.g. the individual green house
        gases included in KYOTOGHG) for only a few points in time.
        From any time points where all the basket contents are known, the relative
        shares of the basket contents are determined, and then interpolated linearly
        and extrapolated constantly to the full timeseries. The shares are then
        used to downscale the basket to its contents, which is used to fill gaps in the
        timeseries of the basket contents.
        Basekt contents are converted to the units of the basket for downscaling, and
        converted back afterwards; for both conversions the gwp conversions of the
        basket are used.

        Parameters
        ----------
        basket: str
          The name of the gas basket for which values are known at higher temporal
          resolution and/or for a wider range. A value from `ds.keys()`.
        basket_contents: list of str
          The name of the gases in the gas basket. The sum of all basket_contents
          equals the basket. Values from `ds.keys()`.
        check_consistency: bool, default True
          If for all points where the basket and all basket_contents are defined,
          it should be checked if the sum of the basket_contents actually equals
          the basket. A ValueError is raised if the consistency check fails.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on `ds.loc[sel]`.
        skip_all_na: bool, default False
          If basket_contents which are NA for all points in a time series are skipped
          entirely.

        Returns
        -------
        downscaled: xr.Dataset
        """
        if sel is not None:
            ds_sel = self._ds.loc[sel]
        else:
            ds_sel = self._ds

        basket_contents_converted = xr.Dataset()
        da_basket = ds_sel[basket]
        for var in basket_contents:
            da: xr.DataArray = ds_sel[var]
            basket_contents_converted[var] = da.pr.convert_to_gwp_like(like=da_basket)

        da_basket_contents: xr.DataArray = basket_contents_converted.to_array("entity")
        if skip_all_na:
            basket_sum = da_basket_contents.pr.sum_skip_all_na(
                dim="entity",
                skipna_evaluation_dims="date",
            )
        else:
            basket_sum = da_basket_contents.sum("entity", skipna=False)

        if check_consistency:
            deviation = abs(da_basket / basket_sum - 1)
            devmax = deviation.max()
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
            .interpolate_na(dim="date", method="linear")
            .ffill(dim="date")
            .bfill(dim="date")
        )

        downscaled = da_basket * shares

        downscaled_converted = xr.Dataset()

        with ureg.context(da_basket.attrs["gwp_context"]):
            for var in basket_contents:
                downscaled_converted[var] = ds_sel[var].fillna(
                    downscaled[var].pint.to(ds_sel[var].pint.units)
                )

        return self._ds.fillna(downscaled_converted)


class DataArrayDownscalingAccessor(BaseDataArrayAccessor):
    def downscale_timeseries(
        self,
        *,
        dimension: str,
        basket: str,
        basket_contents: Sequence[str],
        check_consistency: bool = True,
        sel: Optional[Dict] = None,
        skip_all_na: bool = False,
    ) -> xr.DataArray:
        """Downscale timeseries along a dimension using a basket defined on a
        broader timeseries.

        This is useful if you have data for many points in time for a total, for example
        the entire Energy sector, and higher-resolution data (e.g. fossil and non-fossil
        energies separately) for only a few points in time. In the example, the Energy
        sector is the `basket` and fossil and non-fossil energies are the basket
        contents.
        From any time points where all the basket contents are known, the
        relative shares of the basket contents are determined, and then interpolated
        linearly and extrapolated constantly to the full timeseries. The shares are then
        used to downscale the basket to its contents, which is used to fill gaps in the
        timeseries of the basket contents.

        Parameters
        ----------
        dimension: str
          The name of the dimension which contains the basket and its contents, has to
          be one of the dimensions in `ds.dims`.
        basket: str
          The name of the super-category for which values are known at higher temporal
          resolution and/or for a wider range. A value from `ds[dimension]`.
        basket_contents: list of str
          The name of the sub-categories. The sum of all sub-categories equals the
          basket. Values from `ds[dimension]`.
        check_consistency: bool, default True
          If for all points where the basket and all basket_contents are defined,
          it should be checked if the sum of the basket_contents actually equals
          the basket. A ValueError is raised if the consistency check fails.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on ds.loc[sel].
        skip_all_na: bool, default False
          If basket_contents which are NA for all points in a time series are skipped
          entirely.

        Returns
        -------
        downscaled: xr.DataArray
        """
        if sel is not None:
            da_sel = self._da.loc[sel]
        else:
            da_sel = self._da

        basket_contents_da = da_sel.loc[{dimension: basket_contents}]
        basket_da = da_sel.loc[{dimension: basket}]

        if skip_all_na:
            basket_sum: xr.DataArray = basket_contents_da.pr.sum_skip_all_na(
                dim=dimension, skipna_evaluation_dims="date"
            )
        else:
            basket_sum = basket_contents_da.sum(
                dimension, skipna=False, keep_attrs=True
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
            .interpolate_na(dim="date", method="linear")
            .ffill(dim="date")
            .bfill(dim="date")
        )

        downscaled: xr.DataArray = basket_da * shares

        return self._da.fillna(downscaled)
