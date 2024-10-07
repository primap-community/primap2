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

# Data reading example 3 - minimal test dataset (long)
To run this example the file `test_csv_data_long.csv` must be placed in the same folder as this notebook.
You can find the notebook and the csv file in the folder `docs/source/data_reading` in the PRIMAP2 repository.

```{code-cell} ipython3
# imports
import primap2 as pm2
```

## Dataset Specifications
Here we define which columns of the csv file contain the metadata.
The dict `coords_cols` contains the mapping of csv columns to PRIMAP2 dimensions.
Default values not found in the CSV are set using `coords_defaults`.
The terminologies (e.g. IPCC2006 for categories or the ISO3 country codes for area) are set in the `coords_terminologies` dict.
`coords_value_mapping` defines conversion of metadata values, e.g. category codes.
You can either specify a dict for a metadata column which directly defines the mapping, a function which is used to map metadata values, or a string to select one of the pre-defined functions included in PRIMAP2.
`filter_keep` and `filter_remove` filter the input data.
Each entry in `filter_keep` specifies a subset of the input data which is kept while the subsets defined by `filter_remove` are removed from the input data.

In the example, the CSV contains the coordinates `country`, `category`, `gas`, and `year`.
They are translated into their proper PRIMAP2 names by specifying the in the
`coords_cols` dictionary. Additionally, columns are specified for the `unit`, and
for the actual `data` (which is found in the column `emissions` in the CSV file).
The format used in the `year` column is given using the `time_format` argument.
Values for the `scenario` and `source` coordinate is not available in the csv file;
 therefore, we add them using default values defined in `coords_defaults`.
Terminologies are given for `area`, `category`, `scenario`, and the secondary categories.
Providing these terminologies is mandatory to create a valid PRIMAP2 dataset.

Coordinate mapping is necessary for `category`, `entity`, and `unit`.
They all use the PRIMAP1 specifications in the csv file.
For `category` this means that e.g. `IPC1A2` would be converted to `1.A.2` for `entity` the conversion affects the way GWP information is stored in the entity name: e.g. `KYOTOGHGAR4` is mapped to `KYOTOGHG (AR4GWP100)`.

In this example, we also add `meta_data` to add a reference for the data and usage rights.

```{code-cell} ipython3
file = "test_csv_data_long.csv"
coords_cols = {
    "unit": "unit",
    "entity": "gas",
    "area": "country",
    "category": "category",
    "time": "year",
    "data": "emissions",
}
coords_defaults = {
    "source": "TESTcsv2021",
    "scenario": "HISTORY",
}
coords_terminologies = {
    "area": "ISO3",
    "category": "IPCC2006",
    "scenario": "general",
}
coords_value_mapping = {
    "category": "PRIMAP1",
    "entity": "PRIMAP1",
    "unit": "PRIMAP1",
}
meta_data = {
    "references": "Just ask around.",
    "rights": "public domain",
}
data_if = pm2.pm2io.read_long_csv_file_if(
    file,
    coords_cols=coords_cols,
    coords_defaults=coords_defaults,
    coords_terminologies=coords_terminologies,
    coords_value_mapping=coords_value_mapping,
    meta_data=meta_data,
    time_format="%Y",
)
data_if.head()
```

```{code-cell} ipython3
data_if.attrs
```

## Transformation to PRIMAP2 xarray format
The transformation to PRIMAP2 xarray format is done using the function {meth}`primap2.pm2io.from_interchange_format` which takes an interchange format DataFrame.
The resulting xr Dataset is already quantified, thus the variables are pint arrays which include a unit.

```{code-cell} ipython3
data_pm2 = pm2.pm2io.from_interchange_format(data_if)
data_pm2
```
