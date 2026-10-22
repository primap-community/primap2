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

* `ds` contains emissions of four countries which already carry processing information.
  The CO2 emissions of Colombia are missing for 2002.
* `estimate` contains an expert estimate of the CO2 emissions of Colombia, without
  processing information.

```{code-cell} ipython3
import numpy as np
import xarray as xr

import primap2

ds = primap2.open_dataset("../minimal_ds.nc").pr.loc[{"time": slice("2000", "2005")}]
ds["CO2"].pr.loc[{"area": "COL", "time": "2002"}] = np.nan * primap2.ureg("Gg CO2 / year")
ds = ds.pr.with_creation_step(
    function="open_dataset", description="read from minimal_ds.nc", source="RAND2020"
)

estimate = xr.Dataset(
    {
        "CO2": xr.DataArray(
            np.full((len(ds["time"]), 1, 1), 0.5),
            coords={"time": ds["time"], "area (ISO3)": ["COL"], "source": ["RAND2020"]},
            dims=["time", "area (ISO3)", "source"],
            attrs={"entity": "CO2", "units": "Gg CO2 / year"},
        )
    },
    attrs={"area": "area (ISO3)"},
).pr.quantify()
```

## Reading processing information

{py:meth}`xarray.Dataset.pr.processing_info` returns the processing information of all
data variables as a dataset, with variables named like the data variables, from which
you can select like from the data. The processing information of a single timeseries
is a `ProcessingStepDescription`. Use `format_history()` to show all steps which led to
it, or `history()` to get them as a list:

```{code-cell} ipython3
col = ds.pr.processing_info()["CO2"].pr.loc[{"area": "COL"}]
print(col.item().format_history())
```

Each line starts with the variable and coordinates of the timeseries. The optional data
source of a step is shown as "data source", to tell it apart from the `source`
coordinate. The rest of this page uses a small helper to show histories:

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

## Recording your own processing steps

For processing done with other functions, e.g. directly with xarray, use a
{py:meth}`xarray.Dataset.pr.processing_step` block. Within the block, work on `step.ds`,
a copy of the dataset without processing information, and assign the result to it.
Afterwards, `step.result` contains the result with a processing step for every changed
timeseries. If data is taken from another dataset, pass it as `other_ds`, so that its
processing information becomes part of the history.

```{code-cell} ipython3
with ds.pr.processing_step(
    function="interpolate_na", description="interpolated linearly"
) as step:
    step.ds = step.ds.pr.dequantify().interpolate_na("time").pr.quantify()
show_history(step.result)
```

For your own functions, the {py:func}`primap2.records_processing_step` decorator does
the same. Its first argument is the name of the argument holding the input dataset,
and `{name}` fields in the description are replaced by the arguments of the call. It
also works for methods of dataset accessors with `"self"` as input.

```{code-cell} ipython3
@primap2.records_processing_step("ds", description="scaled by {factor}")
def scale(ds: xr.Dataset, factor: float) -> xr.Dataset:
    return ds * factor


show_history(scale(ds, factor=2), var="SF6")
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

The processing information is selected to the coordinates of the data, so you can also
select from the data before recombining. It has to describe the data as it is, though:
`with_processing_info` can't notice if the data was changed in between. Change data
with primap2 functions, a `processing_step` block or the decorator instead, which
record the change. Usually, there is no need to separate the two at all, as selecting,
saving and all primap2 functions work on datasets with processing information.

## Storing processing information

{py:meth}`xarray.Dataset.pr.to_netcdf` and {py:func}`primap2.open_dataset` save and
read processing information. The interchange format does not support it yet, so use
{py:meth}`xarray.Dataset.pr.without_processing_info` before calling
{py:meth}`xarray.Dataset.pr.to_interchange_format`.
