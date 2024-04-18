import xarray as xr
import numpy as np
import primap2

from scipy.optimize import least_squares
from scipy.linalg import lstsq


class GlobalLSStrategy:
    """Fill missing data in the result dataset by copying.

    The NaNs in the result dataset are substituted with data from the filling
    dataset.
    """

    type = "globalLS"

    def factor_mult(self, a, e, e_ref):
        return a * e - e_ref

    def jac(self, a, e, e_ref):
        J = np.empty((e.size, 1))
        J[:, 0] = e
        return J

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
    ) -> tuple[xr.DataArray, list[primap2.ProcessingStepDescription]]:
        """Fill gaps in ts using data from the fill_ts scaled by a global factor
        defined by global least squares matching. If there is no overlap for least
        squares matching use the SubstitutionStrategy as fallback

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
            data in ts is (partly) filled using scaled data from fill_ts.
            descriptions contains information about which years were affected and
            filled how.
        """

        filled_mask = ts.isnull() & ~fill_ts.isnull()
        time_filled = filled_mask["time"][filled_mask].to_numpy()

        if time_filled.any():
            # check of we have overlap. if not use substitution strategy
            # this might not be necessary because we initialize the LS algorithm with 1,
            # but better make it explicit
            overlap = ts.notnull() & fill_ts.notnull()
            if overlap.any():
                e = fill_ts[overlap.data].data
                e_ref = ts[overlap.data].data
                a0 = [1] # start with 1 as scaling factor
                res = least_squares(self.factor_mult, a0, jac=self.jac, args=(e, e_ref))

                fill_ts_harmo = fill_ts * res['x'][0]
                filled_ts = xr.core.ops.fillna(ts, fill_ts_harmo, join="exact")

                descriptions = [primap2.ProcessingStepDescription(
                    time=time_filled,
                    description="filled with least squares matched data from "
                                f" {fill_ts_repr}. Factor={res['x'][0]:0.3f}",
                    function=self.type,
                    source=fill_ts_repr,
                )]
            else:
                strategy = primap2.csg.SubstitutionStrategy()
                filled_ts, descriptions = strategy.fill(
                    ts=ts,
                    fill_ts=fill_ts,
                    fill_ts_repr=fill_ts_repr,
                )

        else:
            # if we don't have anything to fill we don't need to calculate anything
            filled_ts = ts
            descriptions = [primap2.ProcessingStepDescription(
                time=time_filled,
                description=f"no additional data in {fill_ts_repr}",
                function=self.type,
                source=fill_ts_repr,
            )]

        return filled_ts, descriptions


class GlobalLSlstsqStrategy:
    """Fill missing data in the result dataset by copying.

    The NaNs in the result dataset are substituted with data from the filling
    dataset.
    """

    type = "globalLS_lstsq"

    def fill(
            self,
            *,
            ts: xr.DataArray,
            fill_ts: xr.DataArray,
            fill_ts_repr: str,
    ) -> tuple[xr.DataArray, list[primap2.ProcessingStepDescription]]:
        """Fill gaps in ts using data from the fill_ts scaled by a global factor
        defined by global least squares matching. If there is no overlap for least
        squares matching use the SubstitutionStrategy as fallback

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
            data in ts is (partly) filled using scaled data from fill_ts.
            descriptions contains information about which years were affected and
            filled how.
        """

        filled_mask = ts.isnull() & ~fill_ts.isnull()
        time_filled = filled_mask["time"][filled_mask].to_numpy()

        if time_filled.any():
            # check of we have overlap. if not use substitution strategy
            # this might not be necessary because we initialize the LS algorithm with 1,
            # but better make it explicit
            overlap = ts.notnull() & fill_ts.notnull()
            if overlap.any():
                e = fill_ts[overlap.data].data
                A = np.vstack((e, np.ones_like(e))).transpose()
                e_ref = ts[overlap.data].data
                x, res, rank, s = lstsq(A, e_ref)
                fill_ts_harmo = fill_ts * x[0] + x[1]

                filled_ts = xr.core.ops.fillna(ts, fill_ts_harmo, join="exact")

                descriptions = [primap2.ProcessingStepDescription(
                    time=time_filled,
                    description="filled with least squares matched data from "
                                "{fill_ts_repr}. a*x+b with a={x[0]:0.3f}, "
                                "b={x[1]:0.3f}",
                    function=self.type,
                    source=fill_ts_repr,
                )]
            else:
                strategy = primap2.csg.SubstitutionStrategy()
                filled_ts, descriptions = strategy.fill(
                    ts=ts,
                    fill_ts=fill_ts,
                    fill_ts_repr=fill_ts_repr,
                )

        else:
            # if we don't have anything to fill we don't need to calculate anything
            filled_ts = ts
            descriptions = [primap2.ProcessingStepDescription(
                time=time_filled,
                description=f"no additional data in {fill_ts_repr}",
                function=self.type,
                source=fill_ts_repr,
            )]

        return filled_ts, descriptions
