"""Simple strategy which replaces NaNs by datapoints from second timeseries."""

import attrs
import xarray as xr

from primap2.csg._models import StrategyStep


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
    ) -> tuple[xr.DataArray, list[StrategyStep]]:
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
            filled_ts, steps. filled_ts contains the result, where missing
            data in ts is (partly) filled using unmodified data from fill_ts.
            steps contains information about which years were affected and
            filled how. The filled values are computed from fill_ts at the same time
            point only.
        """
        ts_aligned, fill_ts_aligned = xr.align(ts, fill_ts, join="exact")
        filled_ts = ts_aligned.fillna(fill_ts_aligned)
        filled_mask = ts.isnull() & ~fill_ts.isnull()
        step = StrategyStep(
            function=self.type,
            description=f"substituted with corresponding values from {fill_ts_repr}",
            time=filled_mask["time"][filled_mask].to_numpy(),
        )
        return filled_ts, [step]
