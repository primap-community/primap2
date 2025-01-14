"""Simple strategy which replaces NaNs by datapoints from second timeseries."""

import attrs
import xarray as xr

import primap2


@attrs.define(frozen=True)
class SubstitutionStrategy:
    """Fill missing data in the result dataset by copying.

    The NaNs in the result dataset are substituted with data from the filling
    dataset.
    """

    type = "substitution"

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
    ) -> tuple[xr.DataArray, list[primap2.ProcessingStepDescription]]:
        """Fill gaps in ts using data from the fill_ts.

        Parameters
        ----------
        ts
            Base timeseries. Missing data (NaNs) in this timeseries will be filled.
            This function does not modify the data in ts.
        fill_ts
            Fill timeseries. Data from this timeseries will be used (possibly after
            modification) to fill missing data in the base timeseries.
            This function does not modify the data in fill_ts.
        fill_ts_repr
            String representation of fill_ts. Human-readable short representation of
            the fill_ts (e.g. the source).

        Returns
        -------
            filled_ts, descriptions. filled_ts contains the result, where missing
            data in ts is (partly) filled using unmodified data from fill_ts.
            descriptions contains information about which years were affected and
            filled how.
        """
        filled_ts = xr.core.ops.fillna(ts, fill_ts, join="exact")
        filled_mask = ts.isnull() & ~fill_ts.isnull()
        time_filled = "all" if filled_mask.all() else filled_mask["time"][filled_mask].to_numpy()
        description = primap2.ProcessingStepDescription(
            time=time_filled,
            description=f"substituted with corresponding values from {fill_ts_repr}",
            function=self.type,
            source=fill_ts_repr,
        )
        return filled_ts, [description]
