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

# Add and overwrite data

Generally, datasets in primap2 follow the xarray convention that the data within
datasets is immutable.
To change any data, you need to create a view or a copy of the dataset with the changes
applied.
To this end, we provide a `set` function to set specified data.
It can be used to only fill gaps, add wholly new data, or overwrite existing data in
the dataset.

## The `set` functions

We provide {py:meth}`xarray.DataArray.pr.set` and {py:meth}`xarray.Dataset.pr.set` functions,
for `DataArray`s (individual gases) and `Dataset`s (multiple gases), respectively.

The basic signature of the `set` functions is `set(dimension, keys, values)`, and it
returns the changed object without changing the original one.
Use it like this:

```{code-cell}
# setup: import library and open dataset
import primap2

ds_min = primap2.open_dataset("../minimal_ds.nc")

# Now, select CO2 a slice of the CO2 data as an example to use
da = ds_min["CO2"].loc[{"time": slice("2000", "2005")}]
da
```

```{code-cell}
import numpy as np

from primap2 import ureg

# generate new data for Cuba
new_data_cuba = np.linspace(0, 20, 6) * ureg("Gg CO2 / year")

# Actually modify our original data
modified = da.pr.set("area", "CUB", new_data_cuba)
modified
```

By default, existing non-NaN values are not overwritten:

```{code-cell}
try:
    da.pr.set("area", "COL", np.linspace(0, 20, 6) * ureg("Gg CO2 / year"))
except ValueError as err:
    print(err)
```

You can overwrite existing values by specifying `existing="overwrite"`
to overwrite all values or `existing="fillna"` to overwrite only NaNs.

```{code-cell}
da.pr.set(
    "area",
    "COL",
    np.linspace(0, 20, 6) * ureg("Gg CO2 / year"),
    existing="overwrite",
)
```

By default, the `set()` function extends the specified dimension automatically to
accommodate new values if not all key values are in the specified dimension yet.
You can change this by specifying `new="error"`, which will raise a KeyError if any of
the keys is not found:

```{code-cell}
try:
    da.pr.set(
        "area",
        ["COL", "CUB"],
        np.linspace(0, 20, 6) * ureg("Gg CO2 / year"),
        existing="overwrite",
        new="error",
    )
except KeyError as err:
    print(err)
```

## Example: computing super-categories

In particular, the `set()` functions can also be used with xarray's arithmetic
functions to derive values from existing data and store the result in the Dataset.
As an example, we will derive better values for category 0 by adding all
its subcategories and store the result.

First, let's load a dataset and see the current data for a small subset of the data:

```{code-cell}
ds = primap2.open_dataset("../opulent_ds.nc")

sel = {
    "area": "COL",
    "category": ["0", "1", "2", "3", "4", "5"],
    "animal": "cow",
    "product": "milk",
    "scenario": "highpop",
    "source": "RAND2020",
}
subset = ds.pr.loc[sel].squeeze()

# TODO: currently, plotting with units still emits a warning
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    subset["CO2"].plot.line(x="time", hue="category (IPCC 2006)")
```

While it is hard to see any details in this plot, it is clearly visible
that category 0 is not the sum of the other categories (which should not
come as a surprise because the data were generated at random).

We will now recompute category 0 for the entire dataset using set():

```{code-cell}
cat0_new = ds.pr.loc[{"category": ["1", "2", "3", "4", "5"]}].pr.sum("category")

ds = ds.pr.set(
    "category",
    "0",
    cat0_new,
    existing="overwrite",
)

# plot a small subset of the result
subset = ds.pr.loc[sel].squeeze()
# TODO: currently, plotting with units still emits a warning
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    subset["CO2"].plot.line(x="time", hue="category (IPCC 2006)")
```

As you can see in the plot, category 0 is now computed from its subcategories.
The set() method of Datasets works on all data variables in the dataset which
have the corresponding dimension. In this example, the "population" variable
does not have categories, so it was unchanged.
