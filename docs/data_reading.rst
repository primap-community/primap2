.. highlight:: shell

============
Data reading
============

To work with emissions data in PRIMAP2 it needs to be converted into the
PRIMAP2 netcdf data format. For the most important datasets we will (in
the future) offer datalad packages that can automatically download and
process the data. But currently and for custom data you will need to do
the conversion yourself.


General information
-------------------
The data reading functionality is bundled in the PRIMAP2 submodule ``pm2io``.

To enable a wider use of the PRIMAP2 data reading functionality we read all
data into the PRIMAP2 interchange format which is a wide format pandas
DataFrame with coordinates in columns and following PRIMAP2 specifications.
Additional meta data is stored in ``DataFrame.attrs``. As the ``attrs``
functionality in pandas is experimental it is just stored in the DataFrame
returned by the reading functions and should be stored individually before
doing any processing with the DataFrame.

The PRIMAP2 interchange format can then be converted into native
PRIMAP2 xarray Datasets.

For details on data reading see the following sections and example code linked
therein.

Wide csv file
-------------
The function :meth:`primap2.pm2io.read_wide_csv_file_if` reads wide format csv files
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

.. toctree::
   :maxdepth: 2
   :caption: Examples wide csv:

   data_reading_examples/test_data_wide
   data_reading_examples/PRIMAP-hist

Long (tidy) csv file
--------------------
The function :meth:`primap2.pm2io.read_long_csv_file_if` reads long format CSV files
(also often called tidy CSV files), which are for example used by the FAOstat for
agriculture emissions data.
The input for the function is very similar to the input for
:meth:`primap2.pm2io.read_wide_csv_file_if` described previously, with the difference
mainly that you have to specify the column where to find the data and time information.

To illustrate the use of the function, we have again two examples.
The first example just reads in some example data to understand how the function works.
The second example shows how to read in FAOstat data into PRIMAP2.

.. toctree::
   :maxdepth: 2
   :caption: Examples long CSV:

   data_reading_examples/test_data_long
   data_reading_examples/FAOstat


Further formats
---------------
In the future we will ofer data reading functions for further formats.
Information will be added here.
