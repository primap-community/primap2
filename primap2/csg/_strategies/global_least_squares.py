import numpy as np
import pandas as pd
import xarray as xr
from attrs import field, frozen
from scipy.linalg import lstsq
from scipy.optimize import OptimizeResult, least_squares

import primap2
from primap2.csg._strategies.gaps import MatchParameters

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

    The parameters of the matching and the acceptance of the result are controlled vie the
    `match_params` attribute which is an instance of the `MatchParameters` class.
    The following parameters are available:

    fit_length (default=0) :
        not used in globalLS strategy
    fit_length_unit (default='YS') :
        not used in globalLS strategy
    weighting (default='constant') :
        no other option for globalLS strategy
    var_average_period (default=11) :
        Average period for the rolling mean which is subtracted from the
        time-series before calculating the variance
    allowed_res (default=1) :
        maximal allowed residual of the fit in terms of sigma. The raw residual is
        divided by the number of data points so it can be compared to the variance.
        The allowed residual is max(var(ts), var(fill_ts))*allowed_res**2. The
        comparison is done in terms of variances while allowed_res is given in terms
        of sigma = sqrt(var) for ease of use. The default value of 1 corresponds to
        an allowed residual of one sigma, where `sigma=max(sigma(ts), sigma(fill_ts))`.
        By setting `allowed_res = 0` the check can be disabled.
    allowed_factor (2.01):
        Maximal allowed factor `a` in `fill_ts_harmo=a*fill_ts+b`. Allowed factor
        has to be larger than one as the inverse is used automatically. The default is
        slightly larget than 2 such that a factor of 2 is still allowed and not subject
        to rounding uncertainty.
        By setting `allowed_factor = 0` the check can be disabled.
    allowed_shift (0.201):
        Defines the maximal allowed constant `b` in `fill_ts_harmo=a*fill_ts+b`.
        The allowed shift is defined relative to the mean of fill_ts in the
        period to be filled: `b_max=mean(fill_ts)*allowed_shift.` the default is
        slightly larger than 0.2 such that a shift of 0.2 is still allowed and not subject
        to rounding uncertainty.
        By setting `allowed_shift = 0` the check can be disabled.
    min_var (0.001):
        A minimal assumed variance for comparison with residual. This is necessary
        to avoid failing of the strategy due to rounding errors in the least squares
        matching when variance of a time-series is zero or very small

    Note:
        Very unevenly distributed data points can lead to fit problems,
        e.g. if we use a 10 year period and have 5 data point all in one year. But as the
        normal use case are evenly distributed data points, sometimes with gaps, it's
        currently not relevant


    Attributes
    ----------
    match_params
        Instance of the FitParameters class defining the parameters for the fits on the
        boundaries of the time-series.
    allow_shift: bool, default True
        Allow the filling time series to shift up and down using the additive constant
        :math:`b \\neq 0`.
    allow_negative: bool, default False
        Allow the filling time series to contain negative data initially.
    """

    match_params: MatchParameters = field()
    allow_shift: bool = field()  # True  # check if that works for ruff
    allow_negative: bool = field()
    type = "globalLS"

    @allow_negative.default
    def _set_allow_negative(self):
        return False

    @match_params.default
    def _set_def_match_params(self):
        return MatchParameters(
            fit_length=0,
            fit_length_unit="YS",
            weighting="constant",  # only option currently implemented
            allowed_shift=0,  # keep old behavior
            allowed_factor=0,  # keep old behavior
            allowed_res=0,  # keep old behavior
        )

    @allow_shift.default
    def _set_allow_shift(self):
        return True

    def _factor_mult(self, a, e, e_ref):
        return a * e - e_ref

    def _jac(self, a, e, e_ref):
        J = np.empty((e.size, 1))
        J[:, 0] = e
        return J

    def _variance_detrend(
        self,
        ts: xr.DataArray,
        average_period: int = 11,
    ) -> xr.DataArray:
        """
        Detrend time-series and calculate the variance

        This function first detrends the time-series by subtracting a rolling
        mean using a centered window of length `average_period`. Then the
        variance is calculated and returned. NaN values are ignored in the
        process.

        The ts DataArray can be multi-dimensional. The variance is calculated
        over the time dimension.

        Parameters
        ----------
        ts
            time series to calculate the rolling mean from
        average_period
            length of the averaging period

        Returns
        -------
            xr.DataArray with the variance

        """

        nan_mask = np.isnan(ts)
        average_period = min(average_period, len(ts))
        ts_mean = ts.rolling(time=average_period, min_periods=1, center=True).mean()
        ts_mean[nan_mask] = np.nan
        ts_detrend = ts - ts_mean
        ts_var = ts_detrend.var(dim="time", skipna=True)
        return ts_var

    def _harmo_ls_no_shift(
        self,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        index: pd.DatetimeIndex,
    ) -> tuple[xr.DataArray, OptimizeResult]:
        """
        Harmonize time-series using least squares matching without a shift

        """

        e = fill_ts.loc[{"time": index}].data
        e_ref = ts.loc[{"time": index}].data
        a0 = [1]  #  start with 1 as scaling factor
        result = least_squares(self._factor_mult, a0, jac=self._jac, args=(e, e_ref))
        # check if factor in allowed range
        if self.match_params.allowed_factor != 0:
            if abs(result["x"][0]) > self.match_params.allowed_factor:
                raise StrategyUnableToProcess(
                    reason=f"Harmonization factor a is larger than allowed: "
                    f"a={result['x'][0]:.3f} > "
                    f"{self.match_params.allowed_factor}"
                )
            if abs(result["x"][0]) < 1 / self.match_params.allowed_factor:
                raise StrategyUnableToProcess(
                    reason=f"Harmonization factor a is smaller than allowed: "
                    f"a={result['x'][0]:.3f} < "
                    f"{1 / self.match_params.allowed_factor:.3f}"
                )
        fun = result["fun"]
        res = fun @ fun
        if not np.isnan(res):
            # make residual comparable to sigma
            res = res / len(e_ref)
            # calculate variance of detrended time-series
            var_ts = self._variance_detrend(
                ts.loc[{"time": index}],
                average_period=self.match_params.var_average_period,
            ).values.item()
            var_fill_ts = self._variance_detrend(
                fill_ts.loc[{"time": index}],
                average_period=self.match_params.var_average_period,
            ).values.item()
            max_var = max(var_fill_ts, var_ts, self.match_params.min_var)
            if (res > (max_var * self.match_params.allowed_res**2)) and (
                self.match_params.allowed_res != 0
            ):
                raise StrategyUnableToProcess(
                    reason=f"Residual higher than allowed: "
                    f"sqrt(normed residual)/max(sigma(ts), sigma(fill_ts) "
                    f"= {np.sqrt(res / max_var):.3f} sigma > "
                    f"{self.match_params.allowed_res} sigma."
                )

        fill_ts_harmo = fill_ts * result["x"][0]
        return fill_ts_harmo, result

    def _harmo_ls_shift(
        self,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        index: pd.DatetimeIndex,
    ) -> tuple[xr.DataArray, np.ndarray]:
        """
        Harmonize time-series using least squares matching with a shift

        """
        e = fill_ts.loc[{"time": index}].data
        A = np.vstack((e, np.ones_like(e))).transpose()
        e_ref = ts.loc[{"time": index}].data
        x, res, _rank, _s = lstsq(A, e_ref)
        # check if factor in allowed range
        if self.match_params.allowed_factor != 0:
            if abs(x[0]) > self.match_params.allowed_factor:
                raise StrategyUnableToProcess(
                    reason=f"Harmonization factor a is larger than allowed: "
                    f"a={x[0]:.3f} > {self.match_params.allowed_factor:.3f}"
                )
            if abs(x[0]) < 1 / self.match_params.allowed_factor:
                raise StrategyUnableToProcess(
                    reason=f"Harmonization factor a is smaller than allowed: "
                    f"a={x[0]:.3f} < {1 / self.match_params.allowed_factor:.3f}"
                )
        # check if shift in allowed range
        if (abs(x[1]) > self.match_params.allowed_shift * np.mean(e)) and (
            self.match_params.allowed_shift != 0
        ):
            raise StrategyUnableToProcess(
                reason=f"Harmonization shift a is larger than allowed: "
                f"|b|={abs(x[1]):.3f} > {self.match_params.allowed_shift:.3f}"
            )
        if res.size > 0:
            # make residual comparable to sigma
            res = res / len(e_ref)
            # calculate variance of detrended time-series
            var_ts = self._variance_detrend(
                ts.loc[{"time": index}],
                average_period=self.match_params.var_average_period,
            ).values.item()
            var_fill_ts = self._variance_detrend(
                fill_ts.loc[{"time": index}],
                average_period=self.match_params.var_average_period,
            ).values.item()
            max_var = max(var_fill_ts, var_ts)
            if (res > (max_var * self.match_params.allowed_res**2)) and (
                self.match_params.allowed_res != 0
            ):
                raise StrategyUnableToProcess(
                    reason=f"Residual higher than allowed: "
                    f"sqrt(normed residual)/max(sigma(ts), sigma(fill_ts) "
                    f"= {np.sqrt(res / max_var):.3f} sigma > "
                    f"{self.match_params.allowed_res} sigma."
                )

        fill_ts_harmo = fill_ts * x[0] + x[1]
        if any(fill_ts_harmo < 0):
            # fail, use next strategy
            raise StrategyUnableToProcess(
                reason="Negative data after harmonization excluded by configuration"
            )
        else:
            return fill_ts_harmo, x

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
            index = overlap[overlap].coords["time"]
            if overlap.any():
                if self.allow_shift:
                    fill_ts_harmo, x = self._harmo_ls_shift(
                        ts=ts,
                        fill_ts=fill_ts,
                        index=index,
                    )

                    description = (
                        f"filled with least squares matched data from "
                        f"{fill_ts_repr}. a*x+b with a={x[0]:0.3f}, "
                        f"b={x[1]:0.3f}"
                    )
                else:
                    fill_ts_harmo, result = self._harmo_ls_no_shift(
                        ts=ts,
                        fill_ts=fill_ts,
                        index=index,
                    )

                    description = (
                        "filled with least squares matched data from "
                        f"{fill_ts_repr}. Factor={result['x'][0]:0.3f}"
                    )

                # now actually fill
                # we have to do it gap by gap to not fill gaps where there is no data for
                # the matching

                ts_aligned, fill_ts_aligned = xr.align(ts, fill_ts_harmo, join="exact")
                filled_ts = ts_aligned.fillna(fill_ts_aligned)

                descriptions = [
                    primap2.ProcessingStepDescription(
                        time=time_filled,
                        description=description,
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
