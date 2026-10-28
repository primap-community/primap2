import numpy as np
import xarray as xr
from attrs import field, frozen
from scipy.linalg import lstsq
from scipy.optimize import least_squares

import primap2
from primap2.csg._strategies.gaps import (
    MatchParameters,
    fill_gap,
    get_gaps,
    get_matching_index,
)
from primap2.csg._strategies.lls_functions import (
    variance_detrend,
)

from .exceptions import StrategyUnableToProcess

# TODO
# * add parameter to fail if multiple gaps are present? currently they are filled
# using the same harmonization for all gaps
# optimization: adaptive area LS matching where areas with high overlap and a good fit
# are identified and the matching is done using these areas ignoring other areas
#


@frozen
class LocalLSStrategy:
    """Fill missing data using local least square matching to calculate a scaling factor
    between the time-series.

    The NaNs in the first timeseries :math:`\\textrm{ts}(t)` are filled using harmonized data
    from the lower priority timeseries :math:`\\textrm{fill_ts}(t)`. For harmonization we use

    .. math::

       \\textrm{fill_ts}_h(t) = \\textrm{fill_ts}(t) \\times a + b,

    where :math:`\\textrm{fill_ts}_h(t)` is the harmonized dataset and :math:`a` and :math:`b` are
    determined by minimizing the least squares distance between :math:`\\textrm{ts}(t)`
    and :math:`\\textrm{fill_ts}_h(t)` for the time points selected for matching.

    If the class is initialized with ``allow_shift = True`` the faster
    :py:func:`scipy.linalg.lstsq` function is used and :math:`b` can be arbitrary.
    For the case ``allow_shift = False`` (:math:`b = 0`) :py:func:`scipy.optimize.least_squares`
    is used.

    In contrast to the globalLS strategy this strategy only uses datapoint in the vicinity
    of the gaps and borders for the matching. for each boundary a fit period is calculated.
    The length of the fit period (`match_params.fit_length) is defined in terms of
    `match_params.fit_length_unit`, e.g. years as taken as input by `pd.date_range`.
    Make sure the given parameters fit the data you use.
    The time points used for matching are the points in the union of all fit periods
    which have data for `ts` and `fill_ts`.

    In contrast to the localTrends strategy where each gap or boundary is filled individually,
    we use a global harmonization for `fill_ts` to have a consistent treatment for all gaps and
    boundaries.

    If the set is empty a :py:class:`StrategyUnableToProcess` error will be raised.

    If ``allow_negative = False`` and the harmonized time-series :math:`\\textrm{fill_ts}_h(t)`
    contains negative data a :py:class:`StrategyUnableToProcess` error will be raised.

    The parameters of the matching and the acceptance of the result are controlled vie the
    `match_params` attribute which is an instance of the `MatchParameters` class.
    The following parameters are available:

    fit_length (default=20) :
        length of the trend in time steps (usually years)
    fit_length_unit (default='YS') :
        Unit for the length of the trend. String passed to the `freq` argument of
        `pd.date_range`. Default is 'YS' (yearly at start of year)
    weighting (default='constant') :
        Which weighting to use for the data points. 'constant' uses constant weights for
        all points while 'linear' uses weights that decline linearly from 1 at the
        boundary to 0 at the last point of the fit interval. Currently only 'constant'
        is implemented.
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
    allowed_factor (2.01):
        Maximal allowed factor `a` in `fill_ts_harmo=a*fill_ts+b`. Allowed factor
        has to be larger than one as the inverse is used automatically. The default is
        slightly larget than 2 such that a factor of 2 is still allowed and not subject
        to rounding uncertainty.
    allowed_shift :
        Defines the maximal allowed constant `b` in `fill_ts_harmo=a*fill_ts+b`.
        The allowed shift is defined relative to the mean of fill_ts in the
        period to be filled: `b_max=mean(fill_ts)*allowed_shift.` the default is
        slightly larger than 0.2 such that a shift of 0.2 is still allowed and not subject
        to rounding uncertainty.
    min_var :
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
    allow_negative
        Allow the filling time series to contain negative data initially.
    """

    match_params: MatchParameters = field()
    allow_shift: bool = field()  # True  # check if that works for ruff
    allow_negative: bool = field()
    type = "localLS"

    @allow_negative.default
    def _set_allow_negative(self):
        return False

    @match_params.default
    def _set_def_match_params(self):
        return MatchParameters(
            fit_length=10,
            fit_length_unit="YS",
            weighting="constant",  # only option currently implemented
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

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
    ) -> tuple[xr.DataArray, list[primap2.ProcessingStepDescription]]:
        """Fill missing data using least squares matching with a defined time-span
        on the boundaries.

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
        time_fillable = filled_mask["time"][filled_mask].to_numpy()

        if time_fillable.any():
            # any_filled = False
            time_filled = np.array([], dtype=np.datetime64)
            gaps = get_gaps(ts)

            # boundary period finding code from trend fitting
            index = None
            gaps_to_fill = []
            gaps_not_filled = []
            gap_messages = []
            for gap in gaps:
                gap_description = (
                    f" gap {np.datetime_as_string(gap.left, unit='h')}"
                    f" - {np.datetime_as_string(gap.right, unit='h')}: "
                )
                # check if we have information for the specific gap
                filled_mask_gap = filled_mask.pr.loc[gap.get_date_slice()]
                time_filled_gap = filled_mask_gap["time"][filled_mask_gap].to_numpy()

                if time_filled_gap.any():
                    if gap.type == "gap":
                        # left boundary
                        index_left = get_matching_index(
                            ts,
                            fill_ts=fill_ts,
                            side="left",
                            boundary=gap.left,
                            match_params=self.match_params,
                        )
                        # right boundary
                        index_right = get_matching_index(
                            ts,
                            fill_ts=fill_ts,
                            side="right",
                            boundary=gap.right,
                            match_params=self.match_params,
                        )
                        new_index = index_left.union(index_right)
                    elif gap.type == "end":
                        # left boundary only
                        new_index = get_matching_index(
                            ts,
                            fill_ts=fill_ts,
                            side="left",
                            boundary=gap.left,
                            match_params=self.match_params,
                        )
                    elif gap.type == "start":
                        # right boundary only
                        new_index = get_matching_index(
                            ts,
                            fill_ts=fill_ts,
                            side="right",
                            boundary=gap.right,
                            match_params=self.match_params,
                        )
                    else:
                        # TODO: this error is currently impossible to reach without
                        #  changing the get_gaps function. But we keep it until the
                        #  final structure of the code has been decided
                        raise ValueError(f"Unknown gap type: {gap.type}")

                    if len(new_index) > 0:
                        if index is None:
                            index = new_index
                        else:
                            index = index.union(new_index)
                        gaps_to_fill.append(gap)
                        gap_description += (
                            f"filled for times {np.datetime_as_string(time_filled_gap, unit='h')}; "
                        )
                        time_filled = np.concatenate((time_filled, time_filled_gap))
                    else:
                        gaps_not_filled.append(gap)
                        gap_description += (
                            "Not filled because there is no data overlap in matching period.; "
                        )
                else:
                    gaps_not_filled.append(gap)
                    gap_description += (
                        f"Not filled because there is no data for the gap in {fill_ts_repr}.; "
                    )

                gap_messages.append(gap_description)

            # If we have gaps to fill calculate a transformation of fill_ts using the
            # gaps matching periods
            if len(gaps_to_fill) > 0:
                if self.allow_shift:
                    e = fill_ts.loc[{"time": index}].data
                    A = np.vstack((e, np.ones_like(e))).transpose()
                    e_ref = ts.loc[{"time": index}].data
                    x, res, _rank, _s = lstsq(A, e_ref)
                    # check if factor in allowed range
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
                    if abs(x[1]) > self.match_params.allowed_shift * np.mean(e):
                        raise StrategyUnableToProcess(
                            reason=f"Harmonization shift a is larger than allowed: "
                            f"|b|={abs(x[1]):.3f} > {self.match_params.allowed_shift:.3f}"
                        )
                    if not np.isnan(res):
                        # make residual comparable to sigma
                        res = res / len(e_ref)
                        # calculate variance of detrended time-series
                        var_ts = variance_detrend(
                            ts.loc[{"time": index}],
                            average_period=self.match_params.var_average_period,
                        ).values.item()
                        var_fill_ts = variance_detrend(
                            fill_ts.loc[{"time": index}],
                            average_period=self.match_params.var_average_period,
                        ).values.item()
                        max_var = max(var_fill_ts, var_ts)
                        if res > (max_var * self.match_params.allowed_res**2):
                            raise StrategyUnableToProcess(
                                reason=f"Residual higher than allowed: "
                                f"sqrt(normed residual)/max(sigma(ts), sigma(fill_ts) "
                                f"= {np.sqrt(res / max_var):.3f} sigma > "
                                f"{self.match_params.allowed_res} sigma."
                            )

                    fill_ts_harmo = fill_ts * x[0] + x[1]

                    if any(fill_ts_harmo < 0) and not self.allow_negative:
                        # fail, use next strategy defined in
                        raise StrategyUnableToProcess(
                            reason="Negative data after harmonization excluded by configuration"
                        )
                    else:
                        fill_description = (
                            f"Gaps filled with local least squares matched data "
                            f"from {fill_ts_repr}. a*x+b with a={x[0]:0.3f}, "
                            f"b={x[1]:0.3f}"
                        )

                else:
                    e = fill_ts.loc[{"time": index}].data
                    e_ref = ts.loc[{"time": index}].data
                    a0 = [1]  #  start with 1 as scaling factor
                    result = least_squares(self._factor_mult, a0, jac=self._jac, args=(e, e_ref))
                    # check if factor in allowed range
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
                        var_ts = variance_detrend(
                            ts.loc[{"time": index}],
                            average_period=self.match_params.var_average_period,
                        ).values.item()
                        var_fill_ts = variance_detrend(
                            fill_ts.loc[{"time": index}],
                            average_period=self.match_params.var_average_period,
                        ).values.item()
                        max_var = max(var_fill_ts, var_ts, self.match_params.min_var)
                        if res > (max_var * self.match_params.allowed_res**2):
                            raise StrategyUnableToProcess(
                                reason=f"Residual higher than allowed: "
                                f"sqrt(normed residual)/max(sigma(ts), sigma(fill_ts) "
                                f"= {np.sqrt(res / max_var):.3f} sigma > "
                                f"{self.match_params.allowed_res} sigma."
                            )

                    fill_ts_harmo = fill_ts * result["x"][0]

                    fill_description = (
                        f"Gaps filled with local least squares matched data from "
                        f"{fill_ts_repr}. Factor={result['x'][0]:0.3f}, "
                    )

                # now actually fill
                # we have to do it gap by gap to not fill gaps where there is no data for
                # the matching
                filled_ts = ts.copy()
                for gap in gaps_to_fill:
                    filled_ts = fill_gap(
                        ts=filled_ts, fill_ts=fill_ts_harmo, gap=gap, factor=(1, 1)
                    )

                description = (
                    "The following gaps have been filled:"
                    + "".join(gap_messages)
                    + fill_description
                )

                descriptions = [
                    primap2.ProcessingStepDescription(
                        time=time_filled,
                        description=description,
                        function=self.type,
                        source=fill_ts_repr,
                    )
                ]

            else:
                raise StrategyUnableToProcess(
                    reason="No gap can be filled because there is either no overlap "
                    "for matching or no data to fill the gap."
                )

        else:
            # if we don't have anything to fill we don't need to calculate anything
            filled_ts = ts
            descriptions = [
                primap2.ProcessingStepDescription(
                    time=np.array([], dtype=np.datetime64),
                    description=f"no additional data in {fill_ts_repr}",
                    function=self.type,
                    source=fill_ts_repr,
                )
            ]

        return filled_ts, descriptions
