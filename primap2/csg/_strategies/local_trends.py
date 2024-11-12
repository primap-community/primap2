import xarray as xr
from attrs import frozen

import primap2
from primap2.csg._strategies.gaps import get_gaps

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
    boundary) non-NaN data pint in :math:`\\textrm{ts}`. The trend value is calculated
    using a linear trend of length `trend_length` or less data points if a time-series
    does not cover the full period. By setting `min_trend_points` a minimal number of
    points necessary for the trend calculation can be set. If less points are available a
    :py:class:`StrategyUnableToProcess` error will be raised. This enables the user to
    define a fallback strategy, e.g. single point matching.
    TODO: for the case of gaps this leads to the situation that we can't use trends on
     one side of the gap and single year matching as fallback on the other

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
    Use with case.

    Attributes
    ----------
    trend_length: int, default 1
        Define the number of data point used to calculate trend values for the matching
        point
    min_trend_points: int, default 1
        Minimal number of data points needed for trend calculation. Can't be larger than
        trend_length obviously
    allow_negative: bool, default False
        Allow the filling time series to contain negative data initially.
    """

    allow_negative: bool = False
    trend_length: int = 1
    min_trend_points: int = 1
    type = "localTrends"

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
        time_filled = filled_mask["time"][filled_mask].to_numpy()

        if time_filled.any():
            # check if we have overlap. if not raise a StrategyUnableToProcess error so
            # the user can decide which fallback method to use
            overlap = ts.notnull() & fill_ts.notnull()
            if overlap.any():
                # TODO implement boundary and gap filling
                gaps = get_gaps(ts)
                for gap in gaps:
                    if gap.type == "g":
                        # fill a gap
                        print("not implemented")
                    elif gap.type == "l":
                        # left boundary
                        print("not implemented")
                    elif gap.type == "r":
                        # right boundary
                        print("not implemented")
                    else:
                        raise ValueError(f"Unknown gap type: {gap.type}")

                # e = fill_ts[overlap.data].data
                # e_ref = ts[overlap.data].data
                # a0 = [1]  #  start with 1 as scaling factor
                # res = least_squares(self._factor_mult, a0, jac=self._jac, args=(e, e_ref))
                #
                # fill_ts_h = fill_ts * res["x"][0]
                # filled_ts = xr.core.ops.fillna(ts, fill_ts_h, join="exact")

                descriptions = [
                    primap2.ProcessingStepDescription(
                        time=time_filled,
                        description="filled with local trend matched data from "
                        f"{fill_ts_repr}.",  # TODO add infor for each gap
                        # Factor={res['x'][0]:0.3f}",
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
