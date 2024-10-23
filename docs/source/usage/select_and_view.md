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
# Select and View Data

## Datasets

In PRIMAP2, data is handled in
[xarray datasets](https://xarray.pydata.org/en/stable/data-structures.html#dataset)
with defined dimensions, coordinates and metadata.
If you are not familiar with selecting data from using xarray, we recommend reading
[the corresponding section in xarray's documentation](https://docs.xarray.dev/en/stable/user-guide/indexing.html)
first.

To get going, we will show the most important features of the data format using a
toy example.

```{code-cell} ipython3
import primap2
import primap2.tests

ds = primap2.tests.examples.toy_ds()

ds
```

You can click through the coordinates and variables to check out the contents of the
toy dataset.
As said, primap2 datasets are
[xarray datasets](https://xarray.pydata.org/en/stable/data-structures.html#dataset),
but with clearly defined naming conventions so that the data is self-describing.

Each dataset has a `time` dimension, an `area` dimension and a `source` dimension.
In our toy example, we additionally have a `category` dimension. For the `area` and
`category` dimensions, the *terminology* used for the dimension is given in the
dimension name in braces, e.g. `ISO3` for the area. The terminologies are defined in
the separate [climate-categories](https://climate-categories.readthedocs.io/en/latest/)
package, so that the meaning of the area codes is clearly defined.

In the dataset are data variables. Each greenhouse gas is in a separate data variable,
and if the data variable contains global warming potential equivalent emissions instead
of mass emissions, the used metric is given in braces.

## Selecting

Data can be selected using the
[xarray indexing methods](https://docs.xarray.dev/en/stable/user-guide/indexing.html),
but PRIMAP2 also provides own versions of some of xarray's selection methods
which are easier to use in the primap2 context.

### Getitem
The following selections both select the same:

```{code-cell} ipython3
ds["area (ISO3)"]
```

```{code-cell} ipython3
ds.pr["area"]
```

### The loc Indexer

Similarly, a version of the `loc` indexer is provided which works with the
bare dimension names:

```{code-cell} ipython3
ds.pr.loc[{"time": slice("2016", "2018"), "area": "COL"}]
```

### Negative Selections

Using the primap2 `loc` indexer, you can also use negative selections to select
everything but the specified value or values along a dimension:

```{code-cell} ipython3
from primap2 import Not

ds.pr.loc[{"time": slice("2002", "2005"), "cat": Not(["0", "1", "2"])}]
```

## Metadata

We store metadata about the whole dataset in the `attrs` of the dataset, and
metadata about specific data variables in their respective `attrs`.

```{code-cell} ipython3
ds.attrs
```

```{code-cell} ipython3
ds["CH4 (SARGWP100)"].attrs
```

In our toy example there are only some technical metadata values which are mostly
convenient for e.g. accessing the global warming potential metric without resorting
to string processing. However, you can also add more information, for example a
short description of your dataset in the attribute `title`:

```{code-cell} ipython3
ds.attrs["title"] = "A toy example dataset which contains random data."
```

We have standardized names for a few attributes (e.g. title), which can then also
be accessed via the `pr` namespace:

```{code-cell} ipython3
ds.pr.title
```

You can find the definition of all standardized attributes at
{ref}`data_format_details_dataset_attributes`.

## Unit handling

PRIMAP2 uses the [openscm_units](https://openscm-units.readthedocs.io)
package based on the [Pint](https://pint.readthedocs.io/) library together
with the [pint-xarray](https://pint-xarray.readthedocs.io/en/stable/) library
for handling of units.

### Unit information

To access the unit information, you can use the `pint` accessor on DataArrays provided
by [pint-xarray](https://pint-xarray.readthedocs.io/en/stable/):

```{code-cell} ipython3
ds["CH4"].pint.units
```

### Simple conversions

Simple unit conversions are possible using standard Pint functions:

```{code-cell} ipython3
ch4_kt_per_day = ds["CH4"].pint.to("kt CH4 / day")
ch4_kt_per_day.pint.units
```

### CO2 equivalent units and mass units

To convert mass units (emissions of gases) into global warming potentials in units of
equivalent CO2 emissions, you have to specify a global warming potential context
(also known as global warming potential metric):

```{code-cell} ipython3
ch4_ar4 = ds["CH4"].pr.convert_to_gwp(gwp_context="AR4GWP100", units="Gg CO2 / year")
# The information about the used GWP context is retained:
ch4_ar4.attrs
```

Because the GWP context used for conversion is stored, it is easy to convert back to
mass units:

```{code-cell} ipython3
ch4 = ch4_ar4.pr.convert_to_mass()
ch4.attrs
```

The stored GWP context can also be used to convert another array using the
same context:

```{code-cell} ipython3
ch4_sar = ds["CH4"].pr.convert_to_gwp_like(ds["CH4 (SARGWP100)"])
ch4_sar.attrs
```

### Dropping units

Sometimes, it is necessary or convenient to drop the units, for example to use
arrays as input for external functions which are unit-naive.
This can be done safely by first converting to the target unit, then
dequantifying the dataset or array:

```{code-cell} ipython3
da_nounits = ds["CH4"].pint.to("Mt CH4 / year").pr.dequantify()
da_nounits.attrs
```

Note that the units are then stored in the DataArray's `attrs`, and can be
restored using the {py:meth}`xarray.DataArray.pr.quantify` function.

## Descriptive statistics

To get an overview about the missing information in a Dataset or DataArray, you
can use the {py:meth}`xarray.DataArray.pr.coverage` function. It gives you a summary
of the number of non-NaN data points.

To illustrate this, we use an array with missing information:

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import pandas as pd
import xarray as xr

time = pd.date_range("2000-01-01", "2003-01-01", freq="YS")
area_iso3 = np.array(["COL", "ARG", "MEX"])
category_ipcc = np.array(["1", "2"])
coords = [
    ("category (IPCC2006)", category_ipcc),
    ("area (ISO3)", area_iso3),
    ("time", time),
]
da = xr.DataArray(
    data=[
        [
            [1, 2, 3, 4],
            [np.nan, np.nan, np.nan, np.nan],
            [1, 2, 3, np.nan],
        ],
        [
            [np.nan, 2, np.nan, 4],
            [1, np.nan, 3, np.nan],
            [1, np.nan, 3, np.nan],
        ],
    ],
    coords=coords,
)

da
```

With this array, we can now obtain coverage statistics along given dimensions:

```{code-cell} ipython3
da.pr.coverage("area")
```

```{code-cell} ipython3
da.pr.coverage("time", "area")
```

For Datasets, you can also specify the "entity" as a coordinate:

```{code-cell} ipython3
ds = primap2.tests.examples._cached_opulent_ds.copy(deep=True)
ds["CO2"].pr.loc[{"product": "milk", "area": ["COL", "MEX"]}].pint.magnitude[:] = np.nan

ds.pr.coverage("product", "entity", "area")
```
