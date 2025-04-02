import typing

import numpy as np
import pandas as pd
import xarray as xr
from attrs import field, frozen
from scipy.linalg import lstsq
from scipy.optimize import least_squares

import primap2
from primap2.csg._strategies.gaps import (
    MatchParameters,
    fill_gap,
    get_gaps,
    get_shifted_time_value,
)

from .exceptions import StrategyUnableToProcess

# TODO
# local matching with least squares instead of lineaer trends
# * if more than one gap exists the default behaviour should be to fail because
#   breaking a time-series into parts and use them individually seems hard to justify
#   in most cases as it assumes internal inconsistency of the time-series.
#   However this should be possible when explicitly allowed because some time-series
#   actually are internally inconsistent.


# optimization: adaptive area LS matching where areas with high overlap and a good fit
# are identified and the matching is done using these areas ignoring other areas
#


@frozen
class LocalLSStrategy:
    """Fill missing data using local least square matching to calculate a scaling factor
    between the time-series..

    The NaNs in the first timeseries :math:`\\textrm{ts}(t)` are filled using harmonized data
    from the lower priority timeseries :math:`\\textrm{fill_ts}(t)`.

    For gaps in the data and missing data at the boundaries of time series different
    treatments are used.

    For boundaries the strategy uses

    .. math::

       \\textrm{fill_ts}_h(t) = \\textrm{fill_ts}(t) \\times a,

    where :math:`\\textrm{fill_ts}_h(t)` is the harmonized dataset and :math:`a` is determined
    by

    ---
     TODO: adapt to LLS
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
    ---

    For gaps the left (:math:`t_{bl}`) and right (:math:`t_{br}`) end have to be considered.
    The data is harmonized using

    ---
    TODO: adapt to LLS using the period of both boundary windows
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
    ---

    TODO: filling multiple gaps: two options
     * fail
     * fill with a matching time-span that covers all points from the first to the last
       matching windows

    # TODO: rewrite
    Filling multiple gaps and boundaries with this function is scientifically questionable
    as they will all use different scaling factors and thus don't use a consistent model to
    harmonize one time-series :math:`\\textrm{fill_ts}(t)` to :math:`\\textrm{ts}(t)`.
    Use with care.

    Attributes
    ----------
    match_params
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

    match_params: MatchParameters = field()
    allow_shift: bool = True  # check if that works for ruff
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

    # @allow_shift.default
    # def _set_allow_shift(self):
    #     return True

    def _factor_mult(self, a, e, e_ref):
        return a * e - e_ref

    def _jac(self, a, e, e_ref):
        J = np.empty((e.size, 1))
        J[:, 0] = e
        return J

    def get_matching_index(
        self,
        ts: xr.DataArray,
        side: typing.Literal["left", "right"],
        boundary: np.datetime64,
        match_params: MatchParameters,
    ) -> pd.DatetimeIndex:
        """
        Get index used for matching for the given timeseries, boundary and parameters

        Parameters
        ----------
        ts
            Time series to use as basis for the index
        side
            "left" or "right" side of the gap / boundary
        boundary
            Date value for the boundary (gap side not data side)
        match_params
            Parameters for the matching, e.g. length of matching period

        Returns
        -------
            pd.DatetimeIndex for the matching period

        """
        # left boundary
        point_to_modify = get_shifted_time_value(
            ts, original_value=boundary, shift=1 if side == "right" else -1
        )
        trend_index = pd.date_range(
            start=point_to_modify if side == "right" else None,
            end=point_to_modify if side == "left" else None,
            periods=match_params.fit_length,
            freq=match_params.fit_length_unit,
        )
        trend_index = trend_index.intersection(ts.coords["time"])
        trend_index = trend_index[~ts.pr.loc[{"time": trend_index}].isnull()]
        return trend_index

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

            # TODO: add area between matching regions to matching region?

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
                        index_left = self.get_matching_index(
                            ts, side="left", boundary=gap.left, match_params=self.match_params
                        )
                        # right boundary
                        index_right = self.get_matching_index(
                            ts, side="right", boundary=gap.right, match_params=self.match_params
                        )
                        new_index = index_left.union(index_right)
                    elif gap.type == "end":
                        # left boundary only
                        new_index = self.get_matching_index(
                            ts, side="left", boundary=gap.left, match_params=self.match_params
                        )
                    elif gap.type == "start":
                        # right boundary only
                        new_index = self.get_matching_index(
                            ts, side="right", boundary=gap.right, match_params=self.match_params
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
                        f"Not filled because there is no data for thegap in {fill_ts_repr}.; "
                    )

                gap_messages.append(gap_description)

            # If we have gaps to fill calculate a transformation of fill_ts using the
            # gaps matching periods
            if len(gaps_to_fill) > 0:
                if self.allow_shift:
                    e = fill_ts.loc[{"time": index}].data
                    A = np.vstack((e, np.ones_like(e))).transpose()
                    e_ref = ts.loc[{"time": index}].data
                    x, res, rank, s = lstsq(A, e_ref)
                    fill_ts_harmo = fill_ts * x[0] + x[1]
                    if any(fill_ts_harmo < 0):
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
                    res = least_squares(self._factor_mult, a0, jac=self._jac, args=(e, e_ref))

                    fill_ts_harmo = fill_ts * res["x"][0]

                    fill_description = (
                        f"Gaps filled with local least squares matched data from "
                        f"{fill_ts_repr}. Factor={res['x'][0]:0.3f}, "
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
