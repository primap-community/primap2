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
The data reading functionality is bundled in the PRIMAP2 submodule ``io``.

To enable a wider use of the PRIMAP2 data reading functionality we read all
data into the PRIMAP2 interchange format which is a wide format pandas
DataFrame with metadata in columns and following PRIMAP2 specifications.
The attributes (``attrs``) are stored in ``DataFrame.attrs``. As the ``attrs``
functionality in pandas is experimental it is just stored in the DataFrame
returned by the reading functions and should be stored individually before
doing any processing with the DataFrame.

For use the data in PRIMAP2 the interchange format can be converted into native
PRIMAP2 xarray Datasets.

For details on data reading see the following sections and example code linked
therein.

Wide csv file
-------------
The function ``read_wide_csv_file_if`` reads wide format csv files which are
widely used for emissions data. All metadata columns can be defined using dicts
as input including giving default values for metadata not available in the csv
files. Data can be filtered for wanted or unwanted metadata.

To illustrate the use of the function we have two examples. The first example
illustrates the different input parameters using a simple test dataset while
the second example is a real world use of the function reading the PRIMAP-hist
v2.2 dataset into PRIMAP2.

.. toctree::
   :maxdepth: 2
   :caption: Examples wide csv:

   data_reading_example_test_data
   data_reading_example_PRIMAP-hist



Further formats
---------------
In the future we will ofer data reading functions for long format csv files,
automatic reading of several csv files within a folder and further formats.
Information will be added here.


Creating a datalad package
--------------------------
Once the datalad repository is up and running we will add information on how
to create datalad repositories and how to include them in the PRIMAP2 datalad
repository.
