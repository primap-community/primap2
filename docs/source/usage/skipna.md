---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Dealing with missing information

## Aggregation

xarray provides robust functions for aggregation ({py:meth}`xarray.DataArray.sum`).
PRIMAP2 adds functions which skip missing data points if the
information is missing at all points along certain axes, for example for
a whole time series.
Let's first create an example with missing information:

```{code-cell} ipython3
import pandas as pd
import numpy as np
import xarray as xr
import primap2

time = pd.date_range("2000-01-01", "2003-01-01", freq="YS")
area_iso3 = np.array(["COL", "ARG", "MEX"])
coords = [("area (ISO3)", area_iso3), ("time", time)]
da = xr.DataArray(
    data=[
        [1, 2, 3, 4],
        [np.nan, np.nan, np.nan, np.nan],
        [1, 2, 3, np.nan],
    ],
    coords=coords,
    name="test data"
)

da.pr.to_df()
```

Now, we can use the primap2 {py:meth}`xarray.DataArray.pr.sum` function to evaluate the sum of countries
while ignoring only those countries where the whole timeseries is missing, using the
`skipna_evaluation_dims` parameter:

```{code-cell} ipython3
da.pr.sum(dim="area", skipna_evaluation_dims="time").pr.to_df()
```

If you instead want to skip all NA values, use the `skipna` parameter:

```{code-cell} ipython3
da.pr.sum(dim="area", skipna=True).pr.to_df()
```

```{code-cell} ipython3
# compare this to the result of the standard xarray sum - it also skips NA values by default:

da.sum(dim="area (ISO3)").pr.to_df()
```

## infilling

The same functionality is available for filling in missing information using the
{py:meth}`xarray.DataArray.pr.fill_all_na` function.
In this example, we fill missing information only where the whole time series is missing.

```{code-cell} ipython3
da.pr.fill_all_na("time", value=10).pr.to_df()
```
