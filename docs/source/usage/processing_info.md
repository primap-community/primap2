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

Datasets can record how every data point was derived. This processing
information is stored in data variables named `Processing of {var}`, which hold the
last {py:class}`primap2.ProcessingStep` of each data point. A step refers to the steps
which set the values it was computed from as its `parents`, so the full history of a
data point can be traced back to the original data. The descriptive fields of a step
(function, description, source, and the variable and coordinates of the timeseries) are
stored in a {py:class}`primap2.ProcessingStepDescription`, which is shared by the data
points of a timeseries which were set by the same step. The storage format is described
in the [data format details](../data_format/data_format_details.md).

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
the processing information is stored, renamed to the variables they describe. The
processing information of a single data point is a `ProcessingStep`. Use
`format_history()` to show all steps which led to it, or `history()` to get them as a
list:

```{code-cell} ipython3
col = ds.pr.processing_info()["CO2"].pr.loc[{"area": "COL", "time": "2001"}]
print(col.item().format_history())
```

Each line starts with the variable and coordinates of the timeseries. To see the
history of a whole timeseries, pass its processing information to
{py:func}`primap2.format_timeseries_history`. It shows each description once, with the
time points of the steps which share it, instead of one tree per data point:

```{code-cell} ipython3
col_ts = ds.pr.processing_info()["CO2"].pr.loc[{"area": "COL", "source": "RAND2020"}]
print(primap2.format_timeseries_history(col_ts))
```

The rest of this page uses small helpers to show histories:

```{code-cell} ipython3
def show_history(
    ds: xr.Dataset, var: str = "CO2", area: str = "COL", time: str = "2002"
) -> None:
    info = ds.pr.processing_info()[var].pr.loc[{"area": area, "time": time}]
    print(info.item().format_history())


def show_timeseries_history(ds: xr.Dataset, var: str = "CO2", area: str = "COL") -> None:
    info = ds.pr.processing_info()[var].pr.loc[{"area": area, "source": "RAND2020"}]
    print(primap2.format_timeseries_history(info))
```

## Processing with primap2 functions

primap2 functions which change data record a processing step for every data point
they change, with the data points the result was computed from as parents. Unchanged
data points keep their processing information. The first line of a history is the last
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
appears, later occurrences only show "see (a)". For the whole timeseries, the
aggregate is shown once for all time points:

```{code-cell} ipython3
show_timeseries_history(aggregated, area="LATAM")
```

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
filled = ds.pr.fillna(tracked_estimate)
show_history(filled)
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
step for every changed data point:

```{code-cell} ipython3
@primap2.records_processing_step("ds", description="scaled by {factor}")
def scale(ds: xr.Dataset, factor: float) -> xr.Dataset:
    return ds * factor


show_history(scale(ds, factor=2), var="SF6")
```

The result is matched to the input per data point, by the variable, coordinates and
time:

* Unchanged data points keep their processing information.
* A changed data point gets a new step. Its parent is the step of the old value, as
  the new value is assumed to be computed from the old one, like `scale` does. If the
  new value replaces the old one instead, pass `replaces=True` to the decorator: the
  history of the old value is then dropped. If data is taken from another dataset,
  pass the name of the argument holding it as `other` to the decorator, so that its
  processing information at the same variable, coordinates and time becomes a parent
  as well.
* Data points which didn't have data before get a new step as well, with the parents
  from `other` if there are any.
* Data points changed to missing values lose their processing information.
* The dimensions of the variables can't change. Removed variables and timeseries are
  removed from the processing information.

A step without parents doesn't tell where the values come from. If a changed value
ends up with neither parents nor a `source`, recording raises an error:

```{code-cell} ipython3
@primap2.records_processing_step("ds", description="set to one", replaces=True)
def set_to_one(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    ds["CO2"].pr.loc[{"area": "COL", "time": "2001"}] = 1 * primap2.ureg("Gg CO2 / year")
    return ds


try:
    set_to_one(ds)
except ValueError as err:
    print(err)
```

Pass the `source` of the values, e.g. `source="constant(1)"`, to record them as new data.
In general, `source` names data used by the step which isn't tracked in the dataset.

For processing which doesn't need a function of its own, e.g. a single xarray call, use
a {py:meth}`xarray.Dataset.pr.processing_step` block, which records steps in the same
way. Within the block, work on `step.ds`, a copy of the dataset without processing
information, and assign the result to it. Afterwards, `step.result` contains the result
with its processing information. Data taken from another dataset is passed as
`other_ds`.

```{code-cell} ipython3
with ds.pr.processing_step(function="round", description="rounded to one digit") as step:
    step.ds = step.ds.round(1)
show_history(step.result, time="2001")
```

## Recording values computed from other time points

The decorator and the block link a changed value to the old value and to `other` at the
same coordinates only. Values computed from other data points, e.g. by interpolation or
extrapolation, need to declare them inside the block with
{py:meth}`step.computed_from(var, mask) <primap2.ProcessingStepRecorder.computed_from>`.
The declared data points become parents of the changed values of `var`, in addition to
the old value. Prefer a block without `computed_from` when a value is computed from the
same data point.

The `mask` selects the parents. Dimensions of the mask are named in one of two ways:

* `"parent <dim>"` stands for the dimension `<dim>` of the parent variable, its
  coordinates are labels of the parent, e.g. `"parent time"` or `"parent source"`.
* All other dimensions must be dimensions of `var`: the mask applies to the new values
  with these coordinates. New values with coordinates absent from the mask have no
  parents.

A dimension of the parent variable which the mask doesn't map has the same coordinate as
the new value, which for `time` means the same time point. Mask values of 0, False and
NaN mean "not a parent", all other values (so weights as well) mean "parent". The mask
can be:

* None: the data point at the same coordinates, e.g.
  `step.computed_from("CH4", variable="CO2")` for values computed from another variable.
* a DataArray. If it has a `time` dimension, its slice at the time of each new value is
  used.
* a function which gets the time of the new values and returns the mask for them, without
  `time` dimension. It is only called for time points with changed values. Use it
  instead of a DataArray with `time` and `parent time` dimensions, which takes a lot of
  memory for long timeseries.

The parents are looked up in the dataset before the block, or in `dataset` if given, in
the variable `variable` (default: `var`). Data points without data are ignored. Calls
accumulate.

The first example fills the gap with a linear fit to the whole timeseries. Every known
value of the timeseries is a parent of every new value, so the mask has no `time`
dimension, only `parent time`:

```{code-cell} ipython3
co2 = ds["CO2"]

with ds.pr.processing_step(
    function="interpolate_linear", description="filled gaps with a linear fit"
) as step:
    plain = step.ds["CO2"].pr.dequantify()
    fit = plain.polyfit("time", 1).polyfit_coefficients
    step.ds["CO2"] = plain.fillna(xr.polyval(plain["time"], fit)).pr.quantify()
    step.computed_from("CO2", co2.notnull().rename(time="parent time"))
interpolated = step.result
show_history(interpolated)
```

Only the filled value got a new step, with all other values of the timeseries as
parents. The other values of the timeseries keep their processing information:

```{code-cell} ipython3
show_timeseries_history(interpolated)
```

The second example fills the gap with the last value before it. The parent of a value at
time `t` is the last known value up to `t`, which is the known value with the same
running count of known values as `t`. This mask is computed for each time point by a
function, instead of building a mask over `time` and `parent time`:

```{code-cell} ipython3
p = lambda da: da.rename(time="parent time")
known = co2.notnull()
count = known.cumsum("time")

with ds.pr.processing_step(function="ffill", description="filled gaps with the last value") as step:
    step.ds["CO2"] = step.ds["CO2"].pr.dequantify().ffill("time").pr.quantify()
    step.computed_from("CO2", lambda t: p(known) & (p(count) == count.sel(time=t)))
forward_filled = step.result
show_history(forward_filled)
```

For linear interpolation, the next known value is a parent as well, so the condition
on the count becomes
`(p(count) == count.sel(time=t)) | (p(count) == count.sel(time=t) + 1)`.

The filled value links to the step of the value of 2001. `used_sources` follows the
parents, so the source of the filled value is the source of the data it was computed
from:

```{code-cell} ipython3
used = forward_filled.pr.used_sources()
used["CO2"].pr.loc[{"area": "COL", "source": "RAND2020"}].to_pandas()
```

Other dimensions are declared in the same way. This is a basket sum on a dataset with
categories, where each total depends on its subcategories at the same time:

```python
mask = xr.DataArray(
    [[True, True]],
    dims=["category (IPCC2006)", "parent category (IPCC2006)"],
    coords={"category (IPCC2006)": ["0"], "parent category (IPCC2006)": ["1", "2"]},
)
with ds.pr.processing_step(function="sum", description="summed subcategories") as step:
    step.ds = sum_subcategories(step.ds)
    step.computed_from("CO2", mask)
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

`with_processing_info` only uses the processing information of the data points in the
data, so you can also select from the data, including time, before recombining. It
raises an error if the processing information lacks data points of the data. It has to
describe the data as it is, though:
`with_processing_info` can't notice if the data was changed in between. Change data
with primap2 functions, the decorator or a `processing_step` block instead, which
record the change.

## Storing processing information

{py:meth}`xarray.Dataset.pr.to_netcdf` and {py:func}`primap2.open_dataset` save and
read processing information. Files with processing information written by earlier
development versions can't be read. The interchange format does not support it:
{py:meth}`xarray.Dataset.pr.to_interchange_format` raises an error for datasets with
processing information, use {py:meth}`xarray.Dataset.pr.without_processing_info` first.
{py:meth}`xarray.Dataset.pr.to_df` leaves the processing information out.

## Looking up the sources of data points

{py:meth}`xarray.Dataset.pr.used_sources` determines from the processing information
which sources were used for each data point. The result has an additional dimension
listing the sources, with the value 1 if the source was used and 0 otherwise. The
sources are the `source` of all steps in the history of a data point, i.e. the steps where
data entered. A source counts as used if its step is in the history at any time point.

```{code-cell} ipython3
used = filled.pr.used_sources()
used["CO2"].pr.loc[{"area": "COL", "source": "RAND2020"}].to_pandas()
```

## Plain xarray operations

xarray doesn't know about processing information, so be careful when using its
functions directly on a dataset with processing information:

* Selecting, e.g. with `ds.pr.loc`, `ds.sel` or `ds.isel`, selects the data and its
  processing information alike, also when selecting time points.
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
