import xarray as xr
import numpy as np
import primap2


class NullStrategy:
    """return a nan time-series (used to skip certain fixed coordinate combinations)
    """

    type = "null"


    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
    ) -> tuple[xr.DataArray, list[primap2.ProcessingStepDescription]]:
        """Returns a timeseries of np.nan only. Used to skip certain fixed
        coordinate combinations.
        TODO: implement special treatment such that it is only called once
         or search for this strategy and set all affected time-series to nan
         before running the compose function

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
            filled_ts, descriptions. filled_ts contains np.nan() only
        """

        filled_ts = ts
        filled_ts.loc[:] = np.nan
        descriptions = [primap2.ProcessingStepDescription(
            time="all",
            description=f"fill ts with np.nan instead of data from {fill_ts_repr}",
            function=self.type,
            source="NaN",
        )]

        return filled_ts, descriptions
