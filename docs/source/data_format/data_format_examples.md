---
jupytext:
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

# Data format

In PRIMAP2, data is handled in
[xarray datasets](https://xarray.pydata.org/en/stable/data-structures.html#dataset)
with defined coordinates and metadata.
If you are not familiar with xarray data structures, we recommend reading
[xarray's own primer](https://xarray.pydata.org/en/stable/data-structures.html) first.

Let's start with two examples, one minimal example showing only what is required for a
PRIMAP2 data set, and one opulent example showing the flexibility of the format.

```{code-cell} ipython3
# import all the used libraries
import datetime

import numpy as np
import pandas as pd
import xarray as xr

import primap2
from primap2 import ureg
```

## Minimal example

This example contains only the required metadata, which are the time, the area,
and the source.
It also shows how multiple gases and global warming potentials stemming from the
gases are included in a single dataset and the use of units.

The example is created with dummy data; note that in real usage, you would read
data from a file or API instead.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
time = pd.date_range("2000-01-01", "2020-01-01", freq="YS")
area_iso3 = np.array(["COL", "ARG", "MEX", "BOL"])
minimal = xr.Dataset(
    {
        ent: xr.DataArray(
            data=np.random.default_rng().random((len(time), len(area_iso3), 1)),
            coords={
                "time": time,
                "area (ISO3)": area_iso3,
                "source": ["RAND2020"],
            },
            dims=["time", "area (ISO3)", "source"],
            attrs={"units": f"{ent} Gg / year", "entity": ent},
        )
        for ent in ("CO2", "SF6", "CH4")
    },
    attrs={"area": "area (ISO3)"},
).pr.quantify()

with ureg.context("SARGWP100"):
    minimal["SF6 (SARGWP100)"] = minimal["SF6"].pint.to("CO2 Gg / year")
minimal["SF6 (SARGWP100)"].attrs["gwp_context"] = "SARGWP100"

minimal
```

Explore the dataset by clicking on the icons at the end of the rows, which will
show you the metadata `attrs` and the actual data for each coordinate or variable.

Notice:

* For the time coordinate, python datetime objects are used, and the meaning of
  each data point is therefore directly obvious.
* For the area coordinate, three-letter country abbreviations are used, and their
  meaning is not necessarily obvious. Therefore, the key or name for the area
  coordinate also contains (in parentheses) the used set of categories, here
  ISO-3166 three-letter country abbreviations. To be able to identify the area
  coordinate without parsing strings, the data set `attrs` contain the key-value pair
  `'area': 'area (ISO3)'`, which translates from the simple name to the coordinate
  key including the identifier for the category set.
* The variables all carry an associated `openscm_units` unit. It is the same unit
  for all data points in a variable, but differs between variables because it
  includes the gas.
* The `attrs` of each variable specify the `entity` of the variable. For simple
  gases like the CO2 emissions, this is the same as the variable name, but for
  example for the global warming potential associated with the SF6 emissions,
  it is different.
* When a global warming potential is given, the used conversion factors have to be
  specified explicitly using openscm_units context names, for example
  `SARGWP100` for the global warming potential equivalent factors for a 100-year
  time horizon specified in the second assessment report.

+++

## Opulent example

The opulent example contains every standard metadata and also shows
that the variables in the data set can have a different number of
dimensions. Because it aims to show everything, creating it takes some effort,
skip to the result unless you are interested in the details.

```{code-cell} ipython3
# create with dummy data
coords = {
    "time": pd.date_range("2000-01-01", "2020-01-01", freq="YS"),
    "area (ISO3)": np.array(["COL", "ARG", "MEX", "BOL"]),
    "category (IPCC 2006)": np.array(["0", "1", "2", "3", "4", "5", "1.A", "1.B"]),
    "animal (FAOSTAT)": np.array(["cow", "swine", "goat"]),
    "product (FAOSTAT)": np.array(["milk", "meat"]),
    "scenario (FAOSTAT)": np.array(["highpop", "lowpop"]),
    "provenance": np.array(["projected"]),
    "model": np.array(["FANCYFAO"]),
    "source": np.array(["RAND2020", "RAND2021"]),
}

opulent = xr.Dataset(
    {
        ent: xr.DataArray(
            data=np.random.default_rng().random(tuple(len(x) for x in coords.values())),
            coords=coords,
            dims=list(coords.keys()),
            attrs={"units": f"{ent} Gg / year", "entity": ent},
        )
        for ent in ("CO2", "SF6", "CH4")
    },
    attrs={
        "entity_terminology": "primap2",
        "area": "area (ISO3)",
        "cat": "category (IPCC 2006)",
        "scen": "scenario (FAOSTAT)",
        "references": "doi:10.1012",
        "rights": "Use however you want.",
        "contact": "lol_no_one_will_answer@example.com",
        "title": "Completely invented GHG inventory data",
        "comment": "GHG inventory data ...",
        "institution": "PIK",
        "publication_date": datetime.date(2099, 12, 31),
    },
)

pop_coords = {
    x: coords[x]
    for x in (
        "time",
        "area (ISO3)",
        "provenance",
        "model",
        "source",
    )
}
opulent["population"] = xr.DataArray(
    data=np.random.default_rng().random(tuple(len(x) for x in pop_coords.values())),
    coords=pop_coords,
    dims=list(pop_coords.keys()),
    attrs={"entity": "population", "units": ""},
)

opulent = opulent.assign_coords(
    {
        "category_names": xr.DataArray(
            data=np.array(
                [
                    "total",
                    "industry",
                    "energy",
                    "transportation",
                    "residential",
                    "land use",
                    "heavy industry",
                    "light industry",
                ]
            ),
            coords={"category (IPCC 2006)": coords["category (IPCC 2006)"]},
            dims=["category (IPCC 2006)"],
        )
    }
)

opulent = opulent.pr.quantify()

with ureg.context("SARGWP100"):
    opulent["SF6 (SARGWP100)"] = opulent["SF6"].pint.to("CO2 Gg / year")
opulent["SF6 (SARGWP100)"].attrs["gwp_context"] = "SARGWP100"

opulent
```

Compared to the minimal example, this data set has a lot more to unpack:

* The first thing to notice is that there are a lot more dimensions, in particular
  for the used `model`, the `provenance` of the data, the described `scenario`, and
  the `animal` type. As before, the dimension `scenario`, `animal`, and `product` use a
  specific set of
  categories given in parentheses and with appropriate metadata in the `attrs`.
  The `scenario` is a standard dimension, and the metadata in `attrs` is given using
  the `scen` key. The `animal` and `product` dimensions are nonstandard.
* There is also a coordinate which is not defining a dimension, `category names`. It
  gives additional information about categories, which can be helpful for humans
  trying to make sense of the category codes without looking them up. Note that
  because this coordinate is not used as an index for a dimension, the category
  names do not have to be unique.
* In the data variables, emissions of the different gases all use all dimensions, but
  the population data does not use all dimensions. For each data variable, only the
  dimensions which make sense have to be used.
* In the `attrs`, the terminology for the entities is explicitly defined, so that the
  meaning of the entity attributes is unambiguously defined.
* In the `attrs`, additional metadata useful for humans is included: citable
  `references`, usage `rights`, a descriptive `title`, a long-form `comment`,
   an email address to `contact` for questions about the data set, and the `publication_date`.

+++

## Processing information example

For detailed descriptions of processing steps done to arrive at the dataset at hand, we use rich metadata types.
This example shows a dataset with detailed processing step information.

```{code-cell} ipython3
# we don't actually do the processing, but add corresponding metadata as if we did

time = pd.date_range("2000-01-01", "2020-01-01", freq="YS")
area_iso3 = np.array(["COL", "ARG"])
with_processing = xr.Dataset(
    {
        "CO2": xr.DataArray(
            data=np.random.default_rng().random((len(time), len(area_iso3), 1)),
            coords={
                "time": time,
                "area (ISO3)": area_iso3,
                "source": ["RAND2020"],
            },
            dims=["time", "area (ISO3)", "source"],
            attrs={"units": "CO2 Gg / year", "entity": "CO2"},
        ),
        "Processing of CO2": xr.DataArray(
            data=np.array(
                [
                    [
                        primap2.TimeseriesProcessingDescription(
                            steps=[
                                primap2.ProcessingStepDescription(
                                    time="all",
                                    function="rand",
                                    description="invented from thin air",
                                ),
                                primap2.ProcessingStepDescription(
                                    time=np.array(["2000", "2001"], dtype=np.datetime64),
                                    function="replace",
                                    description="use other data which is also invented, but better",
                                    source="betterData2024",
                                ),
                            ]
                        ),
                        primap2.TimeseriesProcessingDescription(
                            steps=[
                                primap2.ProcessingStepDescription(
                                    time="all",
                                    function="rand",
                                    description="invented from thin air",
                                ),
                            ]
                        ),
                    ]
                ],
                dtype=object,
            ).T,
            dims=["area (ISO3)", "source"],
            attrs={"entity": "Processing of CO2", "described_variable": "CO2"},
        ),
    },
    attrs={"area": "area (ISO3)"},
).pr.quantify()

with_processing
```

Note that the processing information in the data variable "Processing of CO2" has the same dimensions as the
described variable "CO2", with the exception of the "time". The time information is included in the rich
metadata object itself:

```{code-cell} ipython3
print("COL processing:")
print(with_processing["Processing of CO2"].pr.loc[{"area": "COL"}].item())
print()
print("ARG processing:")
print(with_processing["Processing of CO2"].pr.loc[{"area": "ARG"}].item())
```

## Limitations

* xarray does not provide a solution for the management of multiple data sets,
  including search and discovery, change management etc. For this, we use
  [datalad](../datalad).
* At the moment, xarray does not deal with very sparse data efficiently. For large,
  very sparse datasets with lots of dimensions, primap2 is currently not usable.
