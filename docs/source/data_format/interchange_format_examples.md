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

# Interchange format

In PRIMAP2, data is internally handled in
[xarray datasets](https://xarray.pydata.org/en/stable/data-structures.html#dataset)
with defined coordinates and metadata.
On disk this structure is stored as a netcdf file.
Because the netcdf file format was developed for the exchange of multi-dimensional
datasets with a varying number of dimensions for different entities and rich meta data,
we recommend that consumers of datasets published by us use the provided netcdf files.

However, we recognise that many existing workflows rely on tools that handle tabular
data exclusively and therefore also publish in the **PRIMAP2 Interchange Format** which
is a tabular wide format with additional meta data.
Users of the interchange format have to integrate the given meta data carefully into
their workflows to ensure correct results.

## Logical format
In the interchange format all dimensions and time points are represented by columns in
a two-dimensional array.
Values of the time columns are data while values of the other
columns are coordinates.
To store metadata, including the information contained in
the `attrs` dict in the PRIMAP2 xarray format, we use an additional dictionary.
See sections *In-memory representation* and *on-disk representation* below for
information on the storage of these structures.

The requirements for the data, columns, and coordinates follow the requirements
in the standard PRIMAP2 data format.
Dimensions `area` and `source`, which are mandatory in the xarray format, are mandatory
columns in the tabular data in the interchange format.
The `time` dimension is included in the horizontal
dimension of the tabular interchange format. Additionally, we have `unit` and `entity`
as mandatory columns with the restriction that each entity can have only one unit.

All optional dimensions (see [Data format details](data_format_details.md)) can be
added as optional columns. Secondary categories are columns with free format names.
They are listed as secondary columns in the metadata dict.

Column names correspond to the dimension key of the xarray format, i.e. they contain
the terminology in parentheses (e.g. `area (ISO3)`).

The metadata dict has an `attrs` entry, which corresponds to the `attrs` dict of the
xarray format (see [Data format details](data_format_details.md)).
Additionally, the metadata dict contains information on the `dimensions` of the
data for each entity, on the `time_format` of the data columns and (if stored on disk)
on the name of the `data_file`
(see [Interchange format details](interchange_format_details.md)).

## Use
The interchange format is intended for use mainly in two settings.

* To publish data processed using PRIMAP2 in a way that is easy to read by others but
also keeps the internal structure and metadata. The format will be used by future data
publications by the PRIMAP team including PRIMAP-hist.
* To have a common intermediate format for reading data from original sources (mostly
xls or csv files in different formats) to simplify data reading functions and to enable
use of our data reading functionality by other projects. All data is
first read into the interchange format and subsequently converted into the native
PRIMAP2 format. This enables using our data reading routines in other software
packages.

## In-memory representation
The in-memory representation of the interchange format is using a pandas DataFrame
to store the data, and a dict to store the additional metadata. Pandas DataFrames
have the capability to store the metadata on their `attrs`, however this function
is still experimental and subject to change without notice, so care has to be taken
not to lose the data if processing is done on the DataFrame.
For an example see *Examples* section below.

## On-disk representation
On disk the dataset is represented by a csv file containing the array, and a yaml file
containing the additional metadata as a dict.
Both files should have the same name except for the
ending.
On disk, the key `data_file` is added to the metadata dict, which contains the
name of the csv file.
Thus, a function reading interchange format data just needs the yaml
file name to read the data.

## Examples
Here we show a few examples of the interchange format.

```{code-cell} ipython3
# import all the used libraries
import primap2 as pm2
```

### Reading csv data
The PRIMAP2 data reading procedures first convert data into the interchange format.
For explanations of the used parameters see the
[Data reading example](../data_reading/test_data_wide.md). A more complex dataset is
read in [Data reading PRIMAP-hist](../data_reading/old-PRIMAP-hist.md).

```{code-cell} ipython3
file = "../data_reading/test_csv_data_sec_cat.csv"
coords_cols = {
    "unit": "unit",
    "entity": "gas",
    "area": "country",
    "category": "category",
    "sec_cats__Class": "classification",
}
coords_defaults = {
    "source": "TESTcsv2021",
    "sec_cats__Type": "fugitive",
    "scenario": "HISTORY",
}
coords_terminologies = {
    "area": "ISO3",
    "category": "IPCC2006",
    "sec_cats__Type": "type",
    "sec_cats__Class": "class",
    "scenario": "general",
}
coords_value_mapping = {
    "category": "PRIMAP1",
    "entity": "PRIMAP1",
    "unit": "PRIMAP1",
}
data_if = pm2.pm2io.read_wide_csv_file_if(
    file,
    coords_cols=coords_cols,
    coords_defaults=coords_defaults,
    coords_terminologies=coords_terminologies,
    coords_value_mapping=coords_value_mapping,
)
data_if.head()
```

+++

### Writing interchange format data
Data is written using the {py:func}`primap2.pm2io.write_interchange_format` function which takes a filename
and path (`str` or {py:class}`pathlib.Path`), an interchange format dataframe ({py:class}`pandas.DataFrame`)
and optionally an attribute `dict` as inputs. If the filename has an ending, it will be
ignored. The function writes a `yaml` file and a `csv` file.

```{code-cell} ipython3
file_if = "../data_reading/test_csv_data_sec_cat_if"
pm2.pm2io.write_interchange_format(file_if, data_if)
```

### Reading data from disk
To read interchange format data from disk the function {py:func}`primap2.pm2io.read_interchange_format`
is used. It just takes a filename and path as input (`str` or {py:class}`pathlib.Path`) and returns
a {py:class}`pandas.DataFrame` containing the data and metadata. The filename and path has to point
to the `yaml` file. the `csv` file will be read from the filename contained in the `yaml`
file.

```{code-cell} ipython3
data_if_read = pm2.pm2io.read_interchange_format(file_if)
data_if_read.head()
```

+++

### Converting to and from standard PRIMAP2 format
Data in the standard, xarray-based PRIMAP2 format can be converted to and from the interchange format with the corresponding functions:

```{code-cell} ipython3
ds_minimal = pm2.open_dataset("../minimal_ds.nc")

if_minimal = ds_minimal.pr.to_interchange_format()

if_minimal.head()
```

```{code-cell} ipython3
ds_minimal_re = pm2.pm2io.from_interchange_format(if_minimal)

ds_minimal_re
```
