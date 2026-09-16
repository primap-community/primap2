# Data Reading

To work with emissions data in PRIMAP2 it needs to be converted into the
PRIMAP2 netcdf data format. For the most important datasets we will (in
the future) offer datalad packages that can automatically download and
process the data. But currently and for custom data you will need to do
the conversion yourself.

## General information

The data reading functionality is bundled in the PRIMAP2 submodule {ref}`primap2.pm2io`.

To enable a wider use of the PRIMAP2 data reading functionality we read all
data into the PRIMAP2 interchange format which is a wide format pandas
DataFrame with coordinates in columns and following PRIMAP2 specifications.
Additional meta data is stored in `DataFrame.attrs`. As the `attrs`
functionality in pandas is experimental it is just stored in the DataFrame
returned by the reading functions and should be stored individually before
doing any processing with the DataFrame.

The PRIMAP2 interchange format can then be converted into native
PRIMAP2 xarray Datasets.

For details on data reading see the following sections and example code linked
therein.

## Wide csv file

The function {meth}`primap2.pm2io.read_wide_csv_file_if` reads wide format csv files
which are widely used for emissions data.
All coordinate columns can be defined using dicts
as input including giving default values for coordinates not available in the csv
files.
Data can be filtered for wanted or unwanted coordinate values.

To illustrate the use of the function we have two examples.
The first example
illustrates the different input parameters using a simple test dataset while
the second example is a real world use of the function reading the PRIMAP-hist
v2.2 dataset into PRIMAP2.

```{toctree}
:caption: 'Examples wide csv:'
:maxdepth: 2

test_data_wide
old-PRIMAP-hist
```

## Long (tidy) csv file

The function {meth}`primap2.pm2io.read_long_csv_file_if` reads long format CSV files
(also often called tidy CSV files), which are for example used by the FAOstat for
agriculture emissions data.
The input for the function is very similar to the input for
{meth}`primap2.pm2io.read_wide_csv_file_if` described previously, with the difference
mainly that you have to specify the column where to find the data and time information.

To illustrate the use of the function, we have again an example.
The example just reads in some example data to understand how the function works.

```{toctree}
:caption: 'Examples long CSV:'
:maxdepth: 2

test_data_long
```

## Treatment of string codes

String codes like "IE", "NA" etc. need to be mapped to numerical values.
The codes have to be interpreted to select if they have to be mapped to 0 or
NaN. For example "IE" stands for "included elsewhere" and thus it has to be
mapped to 0 to show that emissions in this timeseries are 0 and not missing.

As a default, we use easy rules combined with defined mappings for special cases.
The special cases are checked first, then the code is split at commas and each part is
stripped of dots and whitespace and upper-cased.
The resulting parts are tested against the rules in the same order as below.

- If one of the parts is `FX`, the code is mapped to `np.nan`
- If one of the parts is `IE` and/or `NO`, the code is mapped to 0
- If one of the parts is `NE` and/or `NA` but none is `IE`, `NO`, or `FX`, the code is
  mapped to `np.nan`
- Otherwise, if the code is a number followed by a footnote marker of the form `(X)`,
  the footnote marker is stripped and the number is used
- If none of the rules applies, a `ValueError` is raised

The special cases are

```python
_special_codes = {
    "C": np.nan,
    "CC": np.nan,
    "CH4": np.nan,  # TODO: move to user passed codes in CRT reading
    "nan": np.nan,
    "NaN": np.nan,
    "-": 0,
    "NE0": np.nan,
    "NE(1)": np.nan,
    "": np.nan,
    "FX": np.nan,
}
```

`NaN` and `nan` will be detected as `np.nan`.

Users can define custom rules by assigning a dict in the format of `_special_codes`
to the `convert_str` parameter.

## Further formats

In the future we will offer data reading functions for further formats.
Information will be added here.
