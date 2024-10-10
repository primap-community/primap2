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

# Data reading example 2 - PRIMAP-hist v2.2 #

In this example, we read an old version of PRIMAP-hist which is not available in the
native format because it was produced before the native format was invented.

```{code-cell} ipython3
# imports
import primap2 as pm2
```

## Obtain the input data

The PRIMAP-hist data (doi:10.5281/zenodo.4479172) is available [from Zenodo](https://zenodo.org/record/4479172),
we download it directly.

```{code-cell} ipython3
import requests
response = requests.get("https://zenodo.org/records/4479172/files/PRIMAP-hist_v2.2_19-Jan-2021.csv?download=1")
file = "PRIMAPHIST22__19-Jan-2021.csv"
with open(file, "w") as fd:
    fd.write(response.text)
```

## Dataset Specifications
Here we define which columns of the csv file contain the coordinates.
The dict `coords_cols` contains the mapping of csv columns to PRIMAP2 dimensions.
Default values are set using `coords_defaults`.
The terminologies (e.g. IPCC2006 for categories or the ISO3 country codes for area) are set in the `coords_terminologies` dict.
`coords_value_mapping` defines conversion of metadata values, e.g. category codes.
`filter_keep` and `filter_remove` filter the input data.
Each entry in `filter_keep` specifies a subset of the input data which is kept while the subsets defined by `filter_remove` are removed from the input data.

For details, we refer to the documentation of {py:func}`primap2.pm2io.read_wide_csv_file_if`.

```{code-cell} ipython3
coords_cols = {
    "unit": "unit",
    "entity": "entity",
    "area": "country",
    "scenario": "scenario",
    "category": "category",
}
coords_defaults = {
    "source": "PRIMAP-hist_v2.2",
}
coords_terminologies = {
    "area": "ISO3",
    "category": "IPCC2006",
    "scenario": "PRIMAP-hist",
}

coords_value_mapping = {
    "category": "PRIMAP1",
    "unit": "PRIMAP1",
    "entity": "PRIMAP1",
}

filter_keep = {
    "f1": {
        "entity": "CO2",
        "category": ["IPC2", "IPC1"],
        "country": ["AUS", "BRA", "CHN", "GBR", "AFG"],
    },
    "f2": {
        "entity": "KYOTOGHG",
        "category": ["IPCMAG", "IPC4"],
        "country": ["AUS", "BRA", "CHN", "GBR", "AFG"],
    },
}

filter_remove = {"f1": {"scenario": "HISTTP"}}
# filter_keep = {"f1": {"entity": "KYOTOGHG", "category": ["IPC2", "IPC1"]},}
# filter_keep = {}
# filter_remove = {}

meta_data = {"references": "doi:10.5281/zenodo.4479172"}
```

## Reading the data to interchange format
To enable a wider use of the PRIMAP2 data reading functionality we read into the PRIMAP2 interchange format, which is a wide format pandas DataFrame with coordinates in columns and following PRIMAP2 specifications.
Additional metadata not captured in this format are stored in `DataFrame.attrs` as a dictionary.
As the `attrs` functionality in pandas is experimental it is just stored in the DataFrame returned by the reading functions and should be stored individually before doing any processing with the DataFrame.

Here we read the data using the {meth}`primap2.pm2io.read_wide_csv_file_if` function.
We have specified restrictive filters above to limit the data included in this notebook.

```{code-cell} ipython3
PMH_if = pm2.pm2io.read_wide_csv_file_if(
    file,
    coords_cols=coords_cols,
    coords_defaults=coords_defaults,
    coords_terminologies=coords_terminologies,
    coords_value_mapping=coords_value_mapping,
    filter_keep=filter_keep,
    filter_remove=filter_remove,
    meta_data=meta_data,
)
PMH_if.head()
```

```{code-cell} ipython3
PMH_if.attrs
```

## Transformation to PRIMAP2 xarray format
The transformation to PRIMAP2 xarray format is done using the function {meth}`primap2.pm2io.from_interchange_format` which takes an interchange format DataFrame.
The resulting xr Dataset is already quantified, thus the variables are pint arrays which include a unit.

```{code-cell} ipython3
PMH_pm2 = pm2.pm2io.from_interchange_format(PMH_if)
PMH_pm2
```
