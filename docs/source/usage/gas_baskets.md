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

# Gas baskets

Gas baskets like `KyotoGHG` are essentially the sum of individual emissions. Usually,
gas baskets are specified in units of warming equivalent CO2, so they necessarily
always have to specify a global warming potential metric as well.

We offer a few specialized functions for handling gas baskets.

## Summation

To sum the contents of gas baskets , the function
{py:meth}`xarray.Dataset.pr.gas_basket_contents_sum` is available.

Let's first create an example dataset.

```{code-cell} ipython3
import primap2
import xarray as xr
import numpy as np

# select example dataset
ds = primap2.open_dataset("../minimal_ds.nc").loc[{"time": slice("2000", "2003")}][
    ["CH4", "CO2", "SF6"]
]
ds
```

```{code-cell}
# add (empty) gas basket with corresponding metadata
ds["KyotoGHG (AR4GWP100)"] = xr.full_like(ds["CO2"], np.nan).pr.quantify(units="Gg CO2 / year")
ds["KyotoGHG (AR4GWP100)"].attrs = {"entity": "KyotoGHG", "gwp_context": "AR4GWP100"}

ds
```

Now, we can compute `KyotoGHG` from its contents (assuming for the moment that this
only includes CO$_2$, SF$_6$ and CH$_4$)

```{code-cell}
# compute gas basket from its contents, which have to be given explicitly
ds.pr.gas_basket_contents_sum(
    basket="KyotoGHG (AR4GWP100)",
    basket_contents=["CO2", "SF6", "CH4"],
)
```

Note that like all PRIMAP2 functions,
{py:meth}`xarray.Dataset.pr.gas_basket_contents_sum`
returns the result without overwriting anything in the original dataset,
so you have to explicitly overwrite existing data if you want that:

```{code-cell}
ds["KyotoGHG (AR4GWP100)"] = ds.pr.gas_basket_contents_sum(
    basket="KyotoGHG (AR4GWP100)",
    basket_contents=["CO2", "SF6", "CH4"],
)
```

## Filling in missing information

To fill in missing data in a gas basket, use
{py:meth}`xarray.Dataset.pr.fill_na_gas_basket_from_contents`

```{code-cell}
# delete all data about the years 2002-2003 (inclusive) from the
# KyotoGHG data
ds["KyotoGHG (AR4GWP100)"].loc[{"time": slice("2002", "2003")}].pint.magnitude[:] = np.nan
ds["KyotoGHG (AR4GWP100)"]
```

```{code-cell}
ds.pr.fill_na_gas_basket_from_contents(
    basket="KyotoGHG (AR4GWP100)", basket_contents=["CO2", "SF6", "CH4"]
)
```

The reverse case is that you are missing some data in the timeseries of
individual gases and want to fill those in using downscaled data from
a gas basket.
Here, use
{py:meth}`xarray.Dataset.pr.downscale_gas_timeseries`

```{code-cell}
# delete all data about the years 2005-2009 from the individual gas data
sel = {"time": slice("2002", "2003")}
ds["CO2"].loc[sel].pint.magnitude[:] = np.nan
ds["SF6"].loc[sel].pint.magnitude[:] = np.nan
ds["CH4"].loc[sel].pint.magnitude[:] = np.nan
ds
```

```{code-cell}
# This determines gas shares at the points in time where individual gas
# data is available, interpolates the shares where data is missing, and
# then downscales the gas basket data using the interpolated shares
ds.pr.downscale_gas_timeseries(basket="KyotoGHG (AR4GWP100)", basket_contents=["CO2", "SF6", "CH4"])
```
