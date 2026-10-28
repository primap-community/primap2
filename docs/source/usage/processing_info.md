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

The examples on this page use three inputs:

* `ds` contains made-up emissions of four countries from the source "RAND2020", which
  already carry processing information. The CO2 emissions of Colombia are missing for
  2002 and 2004.
* `estimate` contains an expert estimate of the CO2 emissions of Colombia, without
  processing information.
* `by_category` contains made-up CO2 emissions of Colombia in the categories 1 and 2,
  which carry processing information as well.

```{code-cell} ipython3
import numpy as np
import xarray as xr

import primap2
```

The inputs are created in the following cell, expand it to see the details:

```{code-cell} ipython3
---
mystnb:
  code_prompt_show: Creation of the example data
tags: [hide-cell]
---
from primap2.tests.examples import minimal_ds

# made-up emissions of CO2, SF6 and CH4 for four countries from the source "RAND2020"
ds = minimal_ds().pr.loc[{"time": slice("2000", "2005")}]
# remove the CO2 emissions of Colombia for 2002 and 2004, so that there are gaps to fill
ds["CO2"].pr.loc[{"area": "COL", "time": ["2002", "2004"]}] = np.nan * primap2.ureg(
    "Gg CO2 / year"
)
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

# the same values for every year, in the categories 1 and 2
by_category = xr.Dataset(
    {
        "CO2": xr.DataArray(
            np.full((len(ds["time"]), 1, 2), [0.3, 0.2]),
            coords={
                "time": ds["time"],
                "area (ISO3)": ["COL"],
                "category (IPCC2006)": ["1", "2"],
            },
            dims=["time", "area (ISO3)", "category (IPCC2006)"],
            attrs={"entity": "CO2", "units": "Gg CO2 / year"},
        )
    },
    attrs={"area": "area (ISO3)", "cat": "category (IPCC2006)"},
).pr.quantify()
by_category = by_category.pr.with_creation_step(
    function="by_category", description="made-up example data", source="RAND2020"
)
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

## Recording values computed from other data points

The decorator links a new or changed value only to data points with the same coordinates
and time, in the input and in `other`. Values computed from other data points, e.g. of
other variables or time points, are recorded with a
{py:meth}`xarray.Dataset.pr.processing_step` block instead. Within the block, work on
`step.ds`, a copy of the dataset without processing information, and assign the result
to it. {py:meth}`step.computed_from <primap2.ProcessingStepRecorder.computed_from>`
declares which data points the derived values, i.e. the new and changed values, were
computed from. The declared data points which have data become parents of the derived
values. Afterwards, `step.result` contains the result with its processing information.
Apart from the declared parents, it is recorded like with the decorator:

```{code-cell} ipython3
basket = "KyotoGHG (AR4GWP100)"
gases = ["CO2", "SF6", "CH4"]
with ds.pr.processing_step(
    function="gas_basket_contents_sum", description="summed the gases"
) as step:
    step.ds[basket] = step.ds.pr.gas_basket_contents_sum(basket=basket, basket_contents=gases)
    step.computed_from(basket, variable=gases)
show_history(step.result, var=basket, time="2001")
```

In this example, {py:meth}`xarray.Dataset.pr.gas_basket_contents_sum` returns a
DataArray, so it can't be used on a dataset with processing information directly (see
"Processing with primap2 functions"), but it works on `step.ds`.
`step.computed_from(basket, variable=gases)` declares that the derived values of `basket`
were computed from the values of the `gases` at the same coordinates and time, in the
dataset before the block. Without `computed_from`, the block raises an error, as the
values of the basket would have neither parents nor a source.

As the CO2 of Colombia is missing in 2002, the basket of Colombia in 2002 only has SF6
and CH4 as parents:

```{code-cell} ipython3
show_history(step.result, var=basket)
```

### Parents at other coordinates

If the coordinates of the parents differ from those of the derived value, e.g. for a
value computed from other time points, pass the parents' coordinates as `at`: a
dictionary from dimension to the label or list of labels of the parents along it. Along
dimensions missing in `at`, the parents have the coordinates of the derived value.

Assume we want to fill the gaps with a linear fit to the whole timeseries. All values of
the timeseries with data are parents of the filled values in this case:

```{code-cell} ipython3
with ds.pr.processing_step(
    function="fill_linear_fit", description="filled gaps with a linear fit"
) as step:
    plain = step.ds["CO2"].pr.dequantify()
    fit = plain.polyfit("time", 1).polyfit_coefficients
    step.ds["CO2"] = plain.fillna(xr.polyval(plain["time"], fit)).pr.quantify()
    step.computed_from("CO2", at={"time": ds["time"].values})
fitted = step.result
show_history(fitted)
```

In this example, `at={"time": ds["time"].values}` declares that each derived value was
computed from the values at all time points. As `area (ISO3)` and `source` are missing
in `at`, these coordinates are assumed to be identical to the derived value. Of all
matching coordinatesthem, only those with data become parents, which leaves out the gaps
of 2002 and 2004. Only
the filled values of 2002 and 2004 are changed, so only they get new processing
steps. The other values of the timeseries keep their processing information:

```{code-cell} ipython3
show_timeseries_history(fitted)
```

Other dimensions work the same way, e.g. for summing the categories 1 and 2 of
`by_category` into category 0:

```{code-cell} ipython3
with by_category.pr.processing_step(
    function="add_aggregates_coordinates", description="summed 1 and 2 into 0"
) as step:
    # on a dataset with processing information, add_aggregates_coordinates would record
    # the step on its own
    step.ds = step.ds.pr.add_aggregates_coordinates(
        {"category (IPCC2006)": {"0": {"sources": ["1", "2"]}}}
    )
    step.computed_from("CO2", at={"category (IPCC2006)": ["1", "2"]})
show_history(step.result.pr.loc[{"category": "0"}], time="2001")
```

The values of category 0 are the only derived values. Their parents are the values of
the categories 1 and 2 at the same time and area.

### Labels for each derived value

In the examples above, the labels in `at` are the same for all derived values. Assume
we want to fill the gaps with the last value before them instead. The parent of a filled
value is then the last value with data before it, which differs between the gaps. For
this, `at` also takes a DataArray, which holds the labels of the parents for each
derived value. Here, it holds the time point of the last value with data, shown for the
CO2 of Colombia:

```{code-cell} ipython3
colombia = {"area": "COL", "source": "RAND2020"}
co2 = ds["CO2"]
# the time point of each value with data, NaT for the gaps
known_time = co2["time"].where(co2.notnull())
# the latest time point with data up to each time point
last_known = known_time.cumulative("time").max()
last_known.pr.loc[colombia].to_pandas()
```

For each time point, `last_known` holds the time point of the parent: 2001 for the
filled value of 2002 and 2003 for the filled value of 2004. At time points with data,
it holds the time point itself, but these values are unchanged and get no parents.

With `last_known`, declaring the parents of the filled values is as simple as before:

```{code-cell} ipython3
with ds.pr.processing_step(function="ffill", description="filled gaps with the last value") as step:
    step.ds["CO2"] = step.ds["CO2"].pr.dequantify().ffill("time").pr.quantify()
    step.computed_from("CO2", at={"time": last_known})
forward_filled = step.result
show_history(forward_filled)
show_history(forward_filled, time="2004")
```

A DataArray in `at` is looked up at the coordinates of each derived value, along the
dimensions it shares with the derived variable. In this example, these are `time`,
`area (ISO3)` and `source`: for the filled value of Colombia from RAND2020 in 2002,
`last_known` holds 2001, so its parent is the value of Colombia from RAND2020 in 2001.
A derived value gets no parent from the DataArray if its coordinates are missing in the
DataArray, or if the DataArray holds a missing value (NaT, NaN or None) there.

A DataArray can also list several parents for each derived value, along any dimension
which the derived variable doesn't have. For linear interpolation, the parents are the
last value with data before a gap and the first one after it:

```{code-cell} ipython3
# the earliest time point with data from each time point on, i.e. last_known backwards
next_known = (
    known_time.isel(time=slice(None, None, -1)).cumulative("time").min().sortby("time")
)
neighbours = xr.concat([last_known, next_known], dim="neighbour")
neighbours.pr.loc[colombia].transpose("time", "neighbour").to_pandas()
```

With this coordinate lookup the parents of the derived values can be defined in the same
way as before:

```{code-cell} ipython3
with ds.pr.processing_step(function="interpolate", description="interpolated gaps") as step:
    step.ds["CO2"] = step.ds["CO2"].pr.dequantify().interpolate_na("time").pr.quantify()
    step.computed_from("CO2", at={"time": neighbours})
show_history(step.result, time="2004")
```

Note how in this example, the `neighbour` dimension of the lookup isn't shared by `co2`,
so all of its entries along this dimension become
parents: the filled value of 2004 was computed from the values of 2003 and 2005.

### Parents in other datasets

To look up the parents in another dataset, pass it as `dataset`. It needs to have
processing information for the parent variable:

```{code-cell} ipython3
with ds.pr.processing_step(
    function="fill_scaled", description="filled gaps with the estimate scaled to 2001"
) as step:
    co2 = step.ds["CO2"]
    factor = co2.sel(time="2001-01-01") / estimate["CO2"].sel(time="2001-01-01")
    step.ds["CO2"] = co2.fillna(estimate["CO2"] * factor)
    # ds in 2001, the estimate at the same time and the estimate in 2001
    step.computed_from("CO2", at={"time": "2001"})
    step.computed_from("CO2", dataset=tracked_estimate)
    step.computed_from("CO2", at={"time": "2001"}, dataset=tracked_estimate)
show_history(step.result)
```

In this example, the gaps are filled with the estimate, scaled to match `ds` in 2001.
Each call of `computed_from` adds parents, so the filled value of 2002 is computed from
`ds` in 2001 and from the estimate in 2001 and 2002. The estimate is passed as
`tracked_estimate`, which has processing information. Its CO2 has no `source`
dimension, so the source isn't used to look up the parents. If the other dataset has
other labels along a dimension, e.g. another source, give them in `at`, like
`at={"source": "EXPERT"}`.

### Without computed_from

The block also works without `computed_from`, e.g. for processing which doesn't need a
function of its own, like a single xarray call. Data taken from another dataset at the
same coordinates and time is passed as `other_ds`, like `other` for the decorator:

```{code-cell} ipython3
with ds.pr.processing_step(function="round", description="rounded to one digit") as step:
    step.ds = step.ds.round(1)
show_history(step.result, time="2001")
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
