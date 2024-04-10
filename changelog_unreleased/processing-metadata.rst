* Added metadata variables to the primap2 data format. Metadata variables describe
  the processing steps done to derive the data on a timeseries level. Also added the
  metadata classes used for the description to the public API. We support saving
  datasets with metadata variables to netcdf, but converting to the interchange format
  looses the metadata variables.
