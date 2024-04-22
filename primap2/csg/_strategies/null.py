"""Simple strategy which replaces all data with NaNs."""

import attrs
import numpy as np
import xarray as xr

import primap2


@attrs.define(frozen=True)
class NullStrategy:
    """Replace all data in the timeseries with NaNs.

    Independent of the contents of the initial timeseries and the filling timeseries,
    the resulting timeseries will be all-NaNs. This strategy is always successful.
    The primary use case is to explicitly discard data in all input datasets for
    specific categories because it is known to be wrong or unusable.

    Note that because the null strategy always returns a result, other strategies
    configured after it for the same timeseries will never be called.
    """

    type = "null"

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
    ) -> tuple[xr.DataArray, list[primap2.ProcessingStepDescription]]:
        """Remove all data, return all-NaN timeseries.

        ts
            Base timeseries. Will be used only to determine metadata for the result
            timeseries. Data in it will not be used.
        fill_ts
            Fill timeseries. Unused.
        fill_ts_repr
            String representation of fill_ts. Human-readable short representation of
            the fill_ts (e.g. the source).

        Returns
        -------
            filled_ts, descriptions. filled_ts contains the result, which has the
            same shape and metadata like ts, but contains only NaN.
            descriptions contains a processing description which says that all data
            was disregarded.
        """
        empty_ts = xr.full_like(other=ts, fill_value=np.nan)
        description = primap2.ProcessingStepDescription(
            time="all",
            description=f"filled with NaN values, not using data from {fill_ts_repr}",
            function=self.type,
            source=fill_ts_repr,
        )
        return empty_ts, [description]
