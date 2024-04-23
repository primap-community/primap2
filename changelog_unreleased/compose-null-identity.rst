* Added protocol for skipping strategies when they aren't applicable for
  input timeseries using the StrategyUnableToProcess exception.
* Added NullStrategy() and IdentityStrategy() for invalidating timeseries
  and skipping individual sources in compose(), respectively.
