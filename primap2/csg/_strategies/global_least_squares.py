import numpy as np
import xarray as xr
from attrs import frozen
from scipy.linalg import lstsq
from scipy.optimize import least_squares

import primap2
from primap2.csg import StrategyUnableToProcess


@frozen
class GlobalLSStrategy:
    """Fill missing data by global least square matching.

    The NaNs in the first timeseries `ts(t)` are filled using harmonized data
    from the lower priority timeseries `fill_ts(t)`. For harmonization we use
    fill_ts(t)_h = fill_ts(t) * a + b,
    where fill_ts(t)_h is the harmonized dataset and a and b are determined by minimizing
    the least squares distance between ts(t) and fill_ts(t)_h.

    If the class is initialized with `allow_shift = True` the faster
    `scipy.linalg.lstsq` function is used and b can be arbitrary.
    For the case `allow_shift = False` (b = 0) `scipi.optimize.least_squares` is used.

    If there is no overlap in non-NaN data between ts(t) and fill_ts(t) a
    `StrategyUnableToProcess` error will be raised.

    If `allow_negative = False` and the harmonized time-series fill_ts(t)_h
    contains negative data a `StrategyUnableToProcess` error will be raised.
    """

    allow_shift: bool = True
    allow_negative: bool = False
    type = "globalLS"

    def _factor_mult(self, a, e, e_ref):
        return a * e - e_ref

    def _jac(self, a, e, e_ref):
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
        """Fill missing data by global least square matching.

        For a description of the algorithm, see the documentation of this class.

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
            # check if we have overlap. if not use substitution strategy
            # this might not be necessary because we initialize the LS algorithm with 1,
            # but better make it explicit
            overlap = ts.notnull() & fill_ts.notnull()
            if overlap.any():
                if self.allow_shift:
                    e = fill_ts[overlap.data].data
                    A = np.vstack((e, np.ones_like(e))).transpose()
                    e_ref = ts[overlap.data].data
                    x, res, rank, s = lstsq(A, e_ref)
                    fill_ts_harmo = fill_ts * x[0] + x[1]
                    if any(fill_ts_harmo < 0):
                        # use filling without shift
                        raise StrategyUnableToProcess(
                            reason="Negative data after harmonization excluded "
                            "by configuration"
                        )
                    else:
                        filled_ts = xr.core.ops.fillna(ts, fill_ts_harmo, join="exact")
                        descriptions = [
                            primap2.ProcessingStepDescription(
                                time=time_filled,
                                description=f"filled with least squares matched data from "
                                f"{fill_ts_repr}. a*x+b with a={x[0]:0.3f}, "
                                f"b={x[1]:0.3f}",
                                function=self.type,
                                source=fill_ts_repr,
                            )
                        ]
                else:
                    e = fill_ts[overlap.data].data
                    e_ref = ts[overlap.data].data
                    a0 = [1]  #  start with 1 as scaling factor
                    res = least_squares(
                        self._factor_mult, a0, jac=self._jac, args=(e, e_ref)
                    )

                    fill_ts_h = fill_ts * res["x"][0]
                    filled_ts = xr.core.ops.fillna(ts, fill_ts_h, join="exact")

                    descriptions = [
                        primap2.ProcessingStepDescription(
                            time=time_filled,
                            description="filled with least squares matched data from "
                            f"{fill_ts_repr}. Factor={res['x'][0]:0.3f}",
                            function=self.type,
                            source=fill_ts_repr,
                        )
                    ]
            else:
                raise StrategyUnableToProcess(
                    reason="No overlap between timeseries, can't match"
                )

        else:
            # if we don't have anything to fill we don't need to calculate anything
            filled_ts = ts
            descriptions = [
                primap2.ProcessingStepDescription(
                    time=time_filled,
                    description=f"no additional data in {fill_ts_repr}",
                    function=self.type,
                    source=fill_ts_repr,
                )
            ]

        return filled_ts, descriptions
