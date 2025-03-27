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

# The Composite Source Generator

The {ref}`primap2.csg` module can be used to create a composite dataset from multiple source
datasets using specified rules.

The general strategy for combining datasets is to always treat a single timeseries, i.e.
an array with only the time as dimension.
For each timeseries, the available source timeseries are ordered according to defined
priorities, and the result timeseries is initialized from the highest-priority
timeseries.
Then, lower-priority source timeseries are used in turn to fill any missing information
in the result timeseries, one source timeseries at a time.
For filling the missing information, a strategy (such as direct substitution or
least-squares matching of data) is selected for each source timeseries as configured.
When no missing information is left in the result timeseries, the algorithm terminates.
It also terminates if all source timeseries are used, even if missing information is
left.

## The `compose` function

The core function to use is the {py:func}`primap2.csg.compose` function.
It needs the following input:

* The input dataset, containing all sources. The shape and dimensions of the input dataset also determine the shape
  and dimensions of the composed dataset.
* A definition of priority dimensions and priorities. The priority dimensions are the dimensions in the input dataset
  which will be used to select source datasets. The result dataset will not have the priority dimensions as dimensions
  any more, because along these dimensions, the source timeseries will be combined into a single composite timeseries.
  The priorities are a list of selections which have to specify exactly one value for each priority dimension, so
  that priorities are clearly defined. You can specify values for other dimensions than the priority dimensions, e.g.
  if you want to change the priorities for some countries or categories. You can also specify exclusions from either
  the result or input datasets to skip specific sources or categories.
* A definition of strategies. Using selectors along any input dataset dimensions, it is possible to define filling
  strategies to use. For each timeseries, a filling strategy has to be specified, so it is a good idea to define
  a default filling strategy using an empty selector (see example below).


```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Logging setup for the docs"

# setup logging for the docs - we don't need debug logs
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")
```

```{code-cell} ipython3
import numpy as np
import primap2
import primap2.csg

ureg = primap2.ureg

input_ds = primap2.open_dataset("../opulent_ds.nc")[["CH4", "CO2", "SF6"]]
input_ds["CH4"].loc[
    {
        "category (IPCC 2006)": "1",
        "time": slice("2000", "2001"),
        "scenario (FAOSTAT)": "lowpop",
    }
][:] = np.nan * ureg("Gg CH4 / year")
input_ds
```

```{code-cell}
priority_definition = primap2.csg.PriorityDefinition(
    priority_dimensions=["source", "scenario (FAOSTAT)"],
    priorities=[
        # only applies to category 0: prefer highpop
        {
            "category (IPCC 2006)": "0",
            "source": "RAND2020",
            "scenario (FAOSTAT)": "highpop",
        },
        {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
        {"source": "RAND2020", "scenario (FAOSTAT)": "highpop"},
        {"source": "RAND2021", "scenario (FAOSTAT)": "lowpop"},
        # the RAND2021, highpop combination is not used at all - you don't have to use all source timeseries
    ],
    # category 5 is not defined for CH4 in this example, so we skip processing it
    # altogether
    exclude_result=[{"entity": "CH4", "category (IPCC 2006)": "5"}],
    # in this example, we know that COL has reported wrong data in the RAND2020 source
    # for SF6 category 1, so we exclude it from processing, it will be skipped and the
    # other data sources will be used as configured in the priorities instead.
    exclude_input=[
        {
            "entity": "SF6",
            "category (IPCC 2006)": "1",
            "area (ISO3)": "COL",
            "source": "RAND2020",
        }
    ],
)
```

```{code-cell} ipython3
# We use the empty selector {}, which matches everything, to configure
# to use the substitution strategy for all timeseries.
strategy_definition = primap2.csg.StrategyDefinition(
    strategies=[({}, primap2.csg.SubstitutionStrategy())]
)
```

```{code-cell} ipython3
result_ds = primap2.csg.compose(
    input_data=input_ds,
    priority_definition=priority_definition,
    strategy_definition=strategy_definition,
    progress_bar=None,  # The animated progress bar is useless in the generated documentation

)

result_ds
```

In the result, you can see that the priority dimensions have been removed, and there are
new data variables "Processing of $entity" added which contain detailed information for
each timeseries how it was derived.

```{code-cell}
sel = {
    "animal": "cow",
    "category": ["0", "1"],
    "product": "milk",
    "time": slice("2000", "2002"),
    "area": "MEX",
}
result_ds["CH4"].pr.loc[sel]
```

```{code-cell}
del sel["time"]
result_ds["Processing of CH4"].pr.loc[sel]
```

```{code-cell}
for tpd in result_ds["Processing of CH4"].pr.loc[sel]:
    print(f"category={tpd['category (IPCC 2006)'].item()}")
    print(str(tpd.item()))
    print()
```

We can see that - as configured - for category 0 "highpop" was preferred, and for
category 1 "lowpop" was preferred.
For category 0, the initial timeseries did not contain NaNs, so no filling was needed.
For category 1, there was information missing in the initial timeseries, so the
lower-priority timeseries was used to fill the holes.

## The `create_composite_source` wrapper function

The {py:func}`primap2.csg.compose` function creates a composite time series according to
the given priorities and strategies, but it does not take care of pre- and postprocessing
of the data. It will carry along unnecessary data and the resulting dataset will miss the
priority coordinates. The {py:func}`primap2.csg.create_composite_source` function takes care
of these steps and prepares the input data and completes the output data to a primap2 dataset
with all desired dimensions and metadata.

The function takes the same inputs as {py:func}`primap2.csg.compose` with additional input to
define pre- and postprocessing:

* **result_prio_coords** Defines the vales for the priority coordinates in the output dataset. As the
priority coordinates differ for all input sources there is no canonical value
for the result and it has to be explicitly defined
* **metadata** Set metadata values such as title and references

```{code-cell} ipython3
result_prio_coords = result_prio_coords = {
        "source": {"value": "PRIMAP-test"},
        "scenario": {"value": "HISTORY", "terminology": "PRIMAP"},
    }
metadata = {"references": "test-data", "contact": "test@example.xx"}

```

* **limit_coords** Optional parameter to remove data for coordinate values not needed for the
composition from the input data. The time coordinate is treated separately.
* **time_range** Optional parameter to limit the time coverage of the input data. The input can either be a pandas `DatetimeIndex` or a tuple of `str` or datetime-like in the form (year_from, year_to) where both boundaries are included in the range. Only the overlap of the supplied index or index created from the tuple with the time coordinate of the input dataset will be used.


```{code-cell} ipython3
limit_coords = {'area (ISO3)': ['COL', 'ARG', 'MEX']}
time_range = ("2000", "2010")

```

```{code-cell} ipython3
complete_result_ds = primap2.csg.create_composite_source(
    input_ds,
    priority_definition=priority_definition,
    strategy_definition=strategy_definition,
    result_prio_coords=result_prio_coords,
    limit_coords=limit_coords,
    time_range=time_range,
    metadata=metadata,
    progress_bar=None,
)

complete_result_ds
```


## Filling strategies
Currently the following filling strategies are implemented
* Global least square matching: {py:class}`primap2.csg.GlobalLSStrategy`
* Straight substitution: {py:class}`primap2.csg.SubstitutionStrategy`
* Local trend matching: {py:class}`primap2.csg.LocalTrendsStrategy`
