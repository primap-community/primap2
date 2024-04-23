"""Simple strategy which returns the input unchanged."""

import attrs
import xarray as xr

import primap2


@attrs.define(frozen=True)
class IdentityStrategy:
    """Return the input unchanged.

    Independent of the contents of the filling timeseries, the resulting timeseries will
    be the initial timeseries. This strategy is always successful.
    The primary use case is to explicitly skip a certain source for specific categories
    only because it is known to be wrong or unusable.

    Note that because the identity strategy always returns a result, other strategies
    configured after it for the same input timeseries will never be called.
    """

    type = "identity"

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
    ) -> tuple[xr.DataArray, list[primap2.ProcessingStepDescription]]:
        """Ignore fill_ts, return ts unchanged.

        ts
            Base timeseries. Will be returned unchanged.
        fill_ts
            Fill timeseries. Unused.
        fill_ts_repr
            String representation of fill_ts. Human-readable short representation of
            the fill_ts (e.g. the source).

        Returns
        -------
            filled_ts, descriptions. filled_ts contains the result, which is ts.
            descriptions contains a processing description which says that fill_ts was
            ignored..
        """
        description = primap2.ProcessingStepDescription(
            time="all",
            description=f"skipping {fill_ts_repr}, not using data from it",
            function=self.type,
            source=fill_ts_repr,
        )
        return ts, [description]
