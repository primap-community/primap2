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
```

## Minimal example

This example contains only the required metadata, which are the time, the area,
and the source.
It also shows how multiple gases are included in a single dataset and how units
are used.

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
* The `attrs` of each variable specify the `entity` of the variable. For emissions
  given as a mass, like here, this is the same as the variable name.

+++

## Minimal example in global warming potentials

The same data can also be given as global warming potentials instead of masses:

```{code-cell} ipython3
minimal_in_gwp = minimal.pr.convert_to_gwp(
    gwp_context="SARGWP100", units="CO2 Gg / year"
)

minimal_in_gwp
```

Notice:

* The variables are now named `{entity} ({gwp_context})`, so the variable name
  differs from the `entity`.
* The used conversion factors have to be specified explicitly using openscm_units
  context names in the `gwp_context` attribute, here `SARGWP100` for the global
  warming potential equivalent factors for a 100-year time horizon specified in the
  second assessment report.
* Each single gas is contained only once in a data set, either as a mass or as a
  global warming potential. Gas baskets are exempt from this rule,
  because their composition is unknown so they cannot be converted between global
  warming potentials, and are commonly given in several of them.

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
* There is also a coordinate which is not defining a dimension, `category_names`. It
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

Processing information describes the processing steps done to arrive at the dataset at hand, as `primap2.ProcessingStep` objects, one for each data point.
This example starts a dataset with a creation step and then replaces some values using another source.

```{code-cell} ipython3
time = pd.date_range("2000-01-01", "2004-01-01", freq="YS")
area_iso3 = np.array(["COL", "ARG"])

created = xr.Dataset(
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
    },
    attrs={"area": "area (ISO3)"},
).pr.quantify()
created = created.pr.with_creation_step(
    function="rand", description="invented from thin air", source="RAND2020"
)

# replace the values of Colombia in 2000 and 2001
with created.pr.processing_step(
    function="replace",
    description="use other data which is also invented, but better",
    source="betterData2024",
    replaces=True,
) as step:
    step.ds["CO2"].pr.loc[{"area": "COL", "time": slice("2000", "2001")}] = (
        np.full((2, 1), 2.0) * primap2.ureg("Gg CO2 / year")
    )
with_processing = step.result

with_processing
```

Note that the processing information in the data variable "Processing of CO2" has the same dimensions as the
described variable "CO2". All data points of a timeseries which were set by the same
operation share one `ProcessingStepDescription`:

```{code-cell} ipython3
processing = with_processing.pr.processing_info()["CO2"]
col = processing.pr.loc[{"area": "COL", "source": "RAND2020"}]
print("COL processing in 2000:")
print(col.pr.loc[{"time": "2000"}].item().format_history())
print()
print("COL processing in 2004:")
print(col.pr.loc[{"time": "2004"}].item().format_history())
print()
print("COL processing, all time points:")
print(primap2.format_timeseries_history(col))
```

## Limitations

* xarray does not provide a solution for the management of multiple data sets,
  including search and discovery, change management etc. For this, we use
  [datalad](../datalad).
* At the moment, xarray does not deal with very sparse data efficiently. For large,
  very sparse datasets with lots of dimensions, primap2 is currently not usable.
