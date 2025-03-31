import numpy as np
import xarray as xr
from attrs import field, frozen

import primap2
from primap2.csg._strategies.gaps import FitParameters, calculate_scaling_factor, fill_gap, get_gaps

from .exceptions import StrategyUnableToProcess


@frozen
class LocalTrendsStrategy:
    """Fill missing data using local trends or single overlap values.

    The NaNs in the first timeseries :math:`\\textrm{ts}(t)` are filled using harmonized data
    from the lower priority timeseries :math:`\\textrm{fill_ts}(t)`.

    For gaps in the data and missing data at the boundaries of time series different
    treatments are used.

    For boundaries the strategy uses

    .. math::

       \\textrm{fill_ts}_h(t) = \\textrm{fill_ts}(t) \\times a,

    where :math:`\\textrm{fill_ts}_h(t)` is the harmonized dataset and :math:`a` is determined
    by

    .. math::

        a = \\textrm{fill_ts}_t(t_b) / \\textrm{ts}_t(t_b),

    where :math:`\\textrm{fill_ts}_t(t_b)` is the trend value calculated for
    :math:`\\textrm{fill_ts}(t_b)` and equally for :math:`\\textrm{ts}_t(t_b)`.
    :math:`t_b` is the last (in case of a right boundary) or first (in case of a left
    boundary) non-NaN data point in :math:`\\textrm{ts}`. The trend value is calculated
    using a linear trend of length `trend_length` or less data points if a time-series
    does not cover the full period. By setting `min_trend_points` a minimal number of
    points necessary for the trend calculation can be set. If less points are available a
    :py:class:`StrategyUnableToProcess` error will be raised. This enables the user to
    define a fallback strategy, e.g. single point matching.

    For the case of gaps this leads to the situation that we can't use trends on
    one side of the gap and single year matching as fallback on the other. Left and right
    scaling factors are always calculated using the same method.

    By setting `trend_length` to 1 single year matching is used.

    For gaps the left (:math:`t_{bl}`) and right (:math:`t_{br}`) end have to be considered.
    The data is harmonized using

    .. math::

       \\textrm{fill_ts}_h(t) = \\textrm{fill_ts}(t) \\times
       \\frac{a_l(t_{br}-t)+a_r(t-t_{bl})}{t_{br}-t_{bl}},

    where

    .. math::

        a_l = \\textrm{fill_ts}_t(t_{bl}) / \\textrm{ts}_t(t_{bl}),

    and

    .. math::

        a_r = \\textrm{fill_ts}_t(t_b) / \\textrm{ts}_t(t_b).

    If only one of the ends of the gap has an overlap with :math:`\\textrm{fill_ts}(t)`,
    we use the harmonization factor of this side for the whole gap (we treat the gap like
    a boundary).

    If there is no overlap in non-NaN data between :math:`\\textrm{ts}(t)` and
    :math:`\\textrm{fill_ts}(t)` a :py:class:`StrategyUnableToProcess` error will be raised.

    If ``allow_negative = False`` and the harmonized time-series :math:`\\textrm{fill_ts}_h(t)`
    contains negative data a :py:class:`StrategyUnableToProcess` error will be raised.

    Filling multiple gaps and boundaries with this function is scientifically questionable
    as they will all use different scaling factors and thus don't use a consistent model to
    harmonize one time-series :math:`\\textrm{fill_ts}(t)` to :math:`\\textrm{ts}(t)`.
    Use with care.

    Attributes
    ----------
    fit_params
        Instance of the FitParameters class defining the parameters for the fits on the
        boundaries of the time-series. The default values are
            trend_length=10,  # ten years if default unit for trend length is used)
            min_trend_points=5,   # minimal data points necessary for trend calculation
            trend_length_unit="YS",  # year start datapoint
            fit_degree=1,  # linear trend by default
            fallback_degree=0,  # constant
    allow_negative
        Allow the filling time series to contain negative data initially.
    """

    fit_params: FitParameters = field()
    allow_negative: bool = field()
    type = "localTrends"

    @fit_params.default
    def _set_def_fit_params(self):
        return FitParameters(
            trend_length=10,
            min_trend_points=5,
            trend_length_unit="YS",
            fit_degree=1,  # linear trend by default
            fallback_degree=0,  # take average as fallback
        )

    @allow_negative.default
    def _set_allow_negative(self):
        return False

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
    ) -> tuple[xr.DataArray, list[primap2.ProcessingStepDescription]]:
        """Fill missing data using matching of local trends on the boundaries.

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
            any_filled = False
            time_filled = np.array([], dtype=np.datetime64)
            description = (
                f"filled with local trend matched data from {fill_ts_repr}. "
                f"The following gaps have been filled:"
            )
            gaps = get_gaps(ts)
            filled_ts = ts.copy()
            for gap in gaps:
                gap_description = (
                    f" gap {np.datetime_as_string(gap.left, unit='h')}"
                    f" - {np.datetime_as_string(gap.right, unit='h')}:"
                )
                # check if we have information for the specific gap
                filled_mask_gap = filled_mask.pr.loc[gap.get_date_slice()]
                time_filled_gap = filled_mask_gap["time"][filled_mask_gap].to_numpy()

                if time_filled_gap.any():
                    # get factor
                    factor = calculate_scaling_factor(
                        ts=ts,
                        fill_ts=fill_ts,
                        gap=gap,
                        fit_params=self.fit_params,
                    )
                    # check if positive or negative allowed. if true proceed, if false
                    # use fallback

                    # it would be more consistent to handle the negative value fallback in
                    # calculate_scaling_factor as well. It comes with the drawback that
                    # it can't be controlled from the filling function and that we have
                    # to deal with different return values here

                    if any(factor < 0) and not self.allow_negative:
                        factor = calculate_scaling_factor(
                            ts=ts,
                            fill_ts=fill_ts,
                            gap=gap,
                            fit_params=self.fit_params.get_fallback(),
                        )
                        gap_description = (
                            gap_description + f" negative scaling factor - use fallback degree "
                            f"{self.fit_params.fallback_degree}"
                        )

                    if any(factor < 0) and not self.allow_negative:
                        # negative with fallback. fail to fill gap
                        gap_description = (
                            gap_description
                            + " negative scaling after fallback - failed to fill gap;"
                        )
                    else:
                        if any(np.isnan(factor)):
                            # fail because no factor can be calculated
                            gap_description = (
                                gap_description + " scaling factor is nan - failed to fill gap;"
                            )
                        else:
                            any_filled = True
                            time_filled = np.concatenate((time_filled, time_filled_gap))

                            # fill nans (in the gap only)
                            filled_ts = fill_gap(
                                ts=filled_ts, fill_ts=fill_ts, gap=gap, factor=factor
                            )

                            if factor[0] == factor[1]:
                                gap_description = (
                                    gap_description + f" filled for times "
                                    f"{np.datetime_as_string(time_filled_gap, unit='h')} "
                                    f"using factor {factor[0]:.2f};"
                                )
                            else:
                                gap_description = (
                                    gap_description + f" filled for times "
                                    f"{np.datetime_as_string(time_filled_gap, unit='h')} "
                                    f"using factors {factor[0]:.2f} and {factor[1]:.2f};"
                                )

                    # update description
                    description = description + gap_description

            if any_filled:
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
                    reason="No overlap between timeseries for any gap. Can't match"
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
