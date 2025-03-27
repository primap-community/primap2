import numpy as np
import xarray as xr
from attrs import frozen
from scipy.linalg import lstsq
from scipy.optimize import least_squares

import primap2

from .exceptions import StrategyUnableToProcess


@frozen
class GlobalLSStrategy:
    """Fill missing data by global least square matching.

    The NaNs in the first timeseries :math:`\\textrm{ts}(t)` are filled using harmonized data
    from the lower priority timeseries :math:`\\textrm{fill_ts}(t)`. For harmonization we use

    .. math::

       \\textrm{fill_ts}_h(t) = \\textrm{fill_ts}(t) \\times a + b,

    where :math:`\\textrm{fill_ts}_h(t)` is the harmonized dataset and :math:`a` and :math:`b` are
    determined by minimizing
    the least squares distance between :math:`\\textrm{ts}(t)` and :math:`\\textrm{fill_ts}_h(t)`.

    If the class is initialized with ``allow_shift = True`` the faster
    :py:func:`scipy.linalg.lstsq` function is used and :math:`b` can be arbitrary.
    For the case ``allow_shift = False`` (:math:`b = 0`) :py:func:`scipy.optimize.least_squares`
    is used.

    If there is no overlap in non-NaN data between :math:`\\textrm{ts}(t)` and
    :math:`\\textrm{fill_ts}(t)` a :py:class:`StrategyUnableToProcess` error will be raised.

    If ``allow_negative = False`` and the harmonized time-series :math:`\\textrm{fill_ts}_h(t)`
    contains negative data a :py:class:`StrategyUnableToProcess` error will be raised.

    Attributes
    ----------
    allow_shift: bool, default True
        Allow the filling time series to shift up and down using the additive constant
        :math:`b \\neq 0`.
    allow_negative: bool, default False
        Allow the filling time series to contain negative data initially.
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
            filled_ts, descriptions.
                filled_ts contains the result, where missing
                data in ts is (partly) filled using scaled data from fill_ts.
                descriptions contains information about which years were affected and
                filled how.
        """
        filled_mask = ts.isnull() & ~fill_ts.isnull()
        time_filled = filled_mask["time"][filled_mask].to_numpy()

        if time_filled.any():
            # check if we have overlap. If not raise error so users can define a fallback
            # strategy
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
                            reason="Negative data after harmonization excluded by configuration"
                        )
                    else:
                        ts_aligned, fill_ts_aligned = xr.align(ts, fill_ts_harmo, join="exact")
                        filled_ts = ts_aligned.fillna(fill_ts_aligned)

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
                    res = least_squares(self._factor_mult, a0, jac=self._jac, args=(e, e_ref))

                    fill_ts_h = fill_ts * res["x"][0]

                    ts_aligned, fill_ts_aligned = xr.align(ts, fill_ts_h, join="exact")
                    filled_ts = ts_aligned.fillna(fill_ts_aligned)

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
                raise StrategyUnableToProcess(reason="No overlap between timeseries, can't match")

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
