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

# Downscaling


To downscale a super-category (for example, regional data) to sub-categories
(for example, country-level data in the same region), the
{py:meth}`xarray.DataArray.pr.downscale_timeseries`
function is available. It determines shares from available data points, then
does downscaling for years where full information is not available.

Let's first create an example dataset with regional data and some country data
missing.

```{code-cell} ipython3
---
mystnb:
  code_prompt_show: Logging setup for the docs
tags: [hide-cell]
---
# setup logging for the docs - we don't need debug logs
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")
```

```{code-cell} ipython3
import primap2
import numpy as np
import xarray as xr

# select an example dataset
da = primap2.open_dataset("../minimal_ds.nc")["CO2"].loc[{"time": slice("2000", "2003"), "source": "RAND2020"}]
da.pr.to_df()
```

```{code-cell} ipython3
# compute regional data as sum of country-level data
temp = da.sum(dim="area (ISO3)")
temp = temp.expand_dims({"area (ISO3)": ["LATAM"]})
# delete data from the country level for the years 2002-2003 (inclusive)
da.loc[{"time": slice("2002", "2003")}].pint.magnitude[:] = np.nan
# add regional data to the array
da = xr.concat([da, temp], dim="area (ISO3)")
da.pr.to_df()
```

As you can see, for 2000 and 2001, country-level data is available, but for later
years, only regional ("LATAM") data is available. We now want to extrapolate the
missing data using the shares from early years and the regional data.

```{code-cell} ipython3
# Do the downscaling to fill in country-level data from regional data
da.pr.downscale_timeseries(
    basket="LATAM",
    basket_contents=["BOL", "MEX", "COL", "ARG"],
    dim="area (ISO3)",
)
```

For the downscaling, shares for the countries at the points in time where data for
all countries is available are determined, the shares are inter- and extrapolated where
data is missing, and then the regional data is downscaled using these shares.
