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

# Processing information

Datasets can record for every timeseries how it was derived. This processing
information is stored in data variables named `Processing of {var}`, which hold the
last {py:class}`primap2.ProcessingStepDescription` of each timeseries. Every step
refers to the steps its input came from as its `parents`, so the full history of a
timeseries can be traced back to the original data. Each step also records the variable
and the coordinates of the timeseries it was done on, at the time it was done. The
storage format is described in the
[data format details](../data_format/data_format_details.md).

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

## Example data

The examples on this page use two inputs:

* `ds` contains made-up emissions of four countries from the source "RAND2020", which
  already carry processing information. The CO2 emissions of Colombia are missing for
  2002.
* `estimate` contains an expert estimate of the CO2 emissions of Colombia, without
  processing information.

```{code-cell} ipython3
import numpy as np
import xarray as xr

import primap2
```

```{code-cell} ipython3
---
mystnb:
  code_prompt_show: Creation of the example data
tags: [hide-cell]
---
from primap2.tests.examples import minimal_ds

# made-up emissions of CO2, SF6 and CH4 for four countries from the source "RAND2020"
ds = minimal_ds().pr.loc[{"time": slice("2000", "2005")}]
# remove the CO2 emissions of Colombia for 2002, so that there is a gap to fill
ds["CO2"].pr.loc[{"area": "COL", "time": "2002"}] = np.nan * primap2.ureg("Gg CO2 / year")
# start tracking processing information, see "Adding processing information to data"
ds = ds.pr.with_creation_step(
    function="minimal_ds", description="made-up example data", source="RAND2020"
)

# the "expert" estimate: the same value for every year, only for Colombia
estimate = xr.Dataset(
    {
        "CO2": xr.DataArray(
            np.full((len(ds["time"]), 1), 0.5),
            coords={"time": ds["time"], "area (ISO3)": ["COL"]},
            dims=["time", "area (ISO3)"],
            attrs={"entity": "CO2", "units": "Gg CO2 / year"},
        )
    },
    attrs={"area": "area (ISO3)"},
).pr.quantify()
```

## Reading processing information

{py:meth}`xarray.Dataset.pr.has_processing_info` tells whether a dataset has
processing information. {py:meth}`xarray.Dataset.pr.processing_info` returns it for all
data variables as a dataset: it contains the `Processing of {var}` variables in which
the processing information is stored, renamed to the variables they describe. The processing
information of a single timeseries is a `ProcessingStepDescription`. Use `format_history()`
to show all steps which led to it, or `history()` to get them as a list:

```{code-cell} ipython3
col = ds.pr.processing_info()["CO2"].pr.loc[{"area": "COL"}]
print(col.item().format_history())
```

Each line starts with the variable and coordinates of the timeseries. The rest of this
page uses a small helper to show histories:

```{code-cell} ipython3
def show_history(ds: xr.Dataset, var: str = "CO2", area: str = "COL") -> None:
    print(ds.pr.processing_info()[var].pr.loc[{"area": area}].item().format_history())
```

## Processing with primap2 functions

primap2 functions which change data record a processing step for every timeseries
they change, with the timeseries the result was derived from as parents. Unchanged
timeseries keep their processing information. The first line of a history is the last
step, a step derived from a single parent is followed by that parent, marked with "↑":

```{code-cell} ipython3
show_history(ds.pr.fillna(0))
```

Steps with several parents, like aggregates, branch into one indented line per parent:

```{code-cell} ipython3
aggregated = ds.pr.add_aggregates_coordinates(
    {"area (ISO3)": {"LATAM": {"sources": ["COL", "ARG", "MEX", "BOL"]}}}
)
show_history(aggregated, area="LATAM")
```

A step which is reached via several branches is labelled like "(a)" where it first
appears, later occurrences only show "see (a)".

Functions whose result is not emissions data, like
{py:meth}`xarray.Dataset.pr.coverage` or {py:meth}`xarray.Dataset.pr.count`, ignore
the processing information. Functions which return a DataArray, like
{py:meth}`xarray.Dataset.pr.gas_basket_contents_sum`, can't carry processing
information and raise an error for datasets which have it.

## Adding processing information to data

If a dataset has processing information, data combined with it needs processing
information as well, otherwise the history of the result would be incomplete:

```{code-cell} ipython3
try:
    ds.pr.fillna(estimate)
except ValueError as err:
    print(err)
```

{py:meth}`xarray.Dataset.pr.with_creation_step` adds processing information which
describes the creation of the data, e.g. after reading it.
{py:meth}`xarray.DataArray.pr.with_creation_step` does the same for a DataArray.

```{code-cell} ipython3
tracked_estimate = estimate.pr.with_creation_step(
    function="estimate", description="expert estimate", source="expert"
)
show_history(ds.pr.fillna(tracked_estimate))
```

In the other direction, filling a dataset without processing information with data
which has it gives a result without processing information. To keep the history of the
other data, add processing information to the dataset first.

## Recording your own processing steps

To record processing done with your own functions, decorate them with
{py:func}`primap2.records_processing_step`. Its first argument is the name of the
argument holding the input dataset, and `{name}` fields in the description are replaced
by the arguments of the call. The decorator removes the processing information from the
input before your function is called, so your function only works with the data. It then
adds the processing information to the dataset your function returns, including a new
step for every changed timeseries:

```{code-cell} ipython3
@primap2.records_processing_step("ds", description="scaled by {factor}")
def scale(ds: xr.Dataset, factor: float) -> xr.Dataset:
    return ds * factor


show_history(scale(ds, factor=2), var="SF6")
```

The timeseries of the result are matched to the timeseries of the input by their
variable and coordinates:

* Unchanged timeseries keep their processing information.
* Changed timeseries get a new step, with their previous processing information as
  parent. If data is taken from another dataset, pass the name of the argument
  holding it as `other` to the decorator, so that its processing information at the
  same variable and coordinates becomes a parent as well.
* New variables and timeseries get a new step without parents. This includes renamed
  coordinates and aggregates computed in the function, which therefore lose the history
  of the timeseries they were computed from. Use the primap2 functions for aggregating
  instead, which record all inputs as parents.
* Removed variables and timeseries are removed from the processing information.
* The dimensions of the variables can't change.

For processing which doesn't need a function of its own, e.g. a single xarray call, use
a {py:meth}`xarray.Dataset.pr.processing_step` block, which records steps in the same
way. Within the block, work on `step.ds`, a copy of the dataset without processing
information, and assign the result to it. Afterwards, `step.result` contains the result
with its processing information. Data taken from another dataset is passed as
`other_ds`.

```{code-cell} ipython3
with ds.pr.processing_step(
    function="interpolate_na", description="interpolated linearly"
) as step:
    step.ds = step.ds.pint.interpolate_na("time")
show_history(step.result)
```

## Separating data and processing information

{py:meth}`xarray.Dataset.pr.without_processing_info` returns the data without
processing information, and {py:meth}`xarray.Dataset.pr.with_processing_info` combines
them again:

```{code-cell} ipython3
data = ds.pr.without_processing_info()
processing = ds.pr.processing_info()
recombined = data.pr.with_processing_info(processing)
xr.testing.assert_identical(recombined, ds)
```

`with_processing_info` only uses the processing information of the timeseries in the
data, so you can also select from the data before recombining. It raises an error if
the processing information lacks timeseries of the data. It has to describe the data as
it is, though:
`with_processing_info` can't notice if the data was changed in between. Change data
with primap2 functions, the decorator or a `processing_step` block instead, which
record the change. Usually, there is no need to separate the two at all, as selecting,
saving and all primap2 functions work on datasets with processing information.

## Storing processing information

{py:meth}`xarray.Dataset.pr.to_netcdf` and {py:func}`primap2.open_dataset` save and
read processing information. The interchange format does not support it yet:
{py:meth}`xarray.Dataset.pr.to_interchange_format` raises an error for datasets with
processing information, use {py:meth}`xarray.Dataset.pr.without_processing_info` first.
{py:meth}`xarray.Dataset.pr.to_df` leaves the processing information out.

## Looking up the sources of data points

{py:meth}`xarray.Dataset.pr.used_sources` determines from the processing information
which sources were used for each data point. The result has an additional dimension
listing the sources, with the value 2 if the source's data at the same time point was
used, 1 if its data at other time points might have been used, e.g. for extrapolation, and 0
otherwise. Note that this assignment is only as accurate as the ProcessingStepDescriptions that were created during calculation. For library functions it is exact.

```{code-cell} ipython3
filled = ds.pr.fillna(tracked_estimate)
used = filled.pr.used_sources()
used["CO2"].pr.loc[{"area": "COL"}].squeeze("source", drop=True).to_pandas()
```

The sources are identified by the `source` of the steps which created the data.

## Plain xarray operations

xarray doesn't know about processing information, so be careful when using its
functions directly on a dataset with processing information:

* Selecting, e.g. with `ds.pr.loc`, `ds.sel` or `ds.isel`, selects the data and its
  processing information alike.
* Selecting a single variable, e.g. `ds["CO2"]`, gives a DataArray without its
  processing information. Use `ds.pr.processing_info()["CO2"]` to get it.
* Selecting several variables with `ds[["CO2", "CH4"]]` drops their processing
  information, use `ds.pr[["CO2", "CH4"]]` instead.
* Calculations on the whole dataset, e.g. `ds * 2`, fail, because they are also applied
  to the processing information.
* Changing data in other ways, e.g. with `ds.fillna(...)`, by assigning
  `ds["CO2"] = ...` or with `xr.merge`, does not update the processing information,
  which then silently describes the data before the change. Use the primap2 functions,
  e.g. `ds.pr.fillna` or `ds.pr.merge`, or record the change as described above.
* Saving with `ds.to_netcdf` fails, use `ds.pr.to_netcdf` instead.
