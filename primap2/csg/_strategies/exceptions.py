class StrategyUnableToProcess(Exception):
    """The filling strategy is unable to process the given timeseries, possibly due
    to missing data.
    """

    def __init__(self, reason: str):
        """Specify the reason why the filling strategy is unable to process the data."""
        self.reason = reason
