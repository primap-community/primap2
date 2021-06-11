==========================
Interchange format details
==========================

The interchange format consists of a wide tabular data object and an additional
dictionary carrying the meta data.

In memory, the tabular data object is a pandas DataFrame and the meta data object
is a python dictionary.
For storage, the tabular data is written to a CSV file and the meta data is written
to a YAML file.

Tabular data
------------

The data is stored in a wide format.
Each row is a time series.
The columns list first all coordinate values for the time series, then the time points.
An example table representation is:

===========  ===================  ================  ===============  ====  ====  ====  ====
area (ISO3)  category (IPCC2006)  entity (primap2)  unit             2000  2001  2002  2003
-----------  -------------------  ----------------  ---------------  ----  ----  ----  ----
"COL"        "1"                  "CO2"             "Gg CO2 / year"  2.3   2.2   2.0   1.9
"COL"        "2"                  "CO2"             "Gg CO2 / year"  1.5   1.6   1.3   1.2
===========  ===================  ================  ===============  ====  ====  ====  ====

Specifically, the columns consist of:

* All dimensions except the ``time`` defined on the Dataset as defined in
  :ref:`data_format_details`,
  including the category set (terminology) in brackets as in the standard data format.
* The entity (with its terminology in brackets, if an entity terminology is defined
  for the dataset), which is used to store the data variable name. The full variable
  name including the global warming potential if applicable is used here.
* The unit in a format which can be parsed by openscm-units.
* One column per value in the ``time`` dimension of the Dataset, formatted according
  to the ``time_format`` strftime format string given in the meta data (see below).

The strictly tabular data format makes it possible to read the data e.g. into Excel,
but imposes multiple inefficiencies:

* In PRIMAP2 data sets, the unit is identical for all time series of the same entity.
  Still, the unit is stored for each time series.
* In PRIMAP2 data sets, not all entities use all dimensions. For example, population
  data might be given together with emissions data, but only the emissions data use
  categories. However, the tabular format imposes to store all entities with the same
  dimensions. Therefore, the dimensions that each entity uses are listed in the
  meta data (see below) and dimensions which are not used for the entity are denoted
  with an empty string in the tabular data.

Meta Data
---------

To correctly interpret the tabular data, meta data is necessary.
The meta data is a dictionary with the following keys:

======================  =========  ================================================================================
key                     data type   meaning
----------------------  ---------  --------------------------------------------------------------------------------
attrs                   dict       The ``attrs`` dictionary of the dataset as defined in :ref:`data_format_details`
data_file               str        The relative path to the CSV data file (only when stored, not in-memory)
dimensions              dict       Mapping of the entities to a list of the dimensions used by them
time_format             str        strftime style time format string used for the time columns
additional_coordinates  dict       Mapping of additional coordinate entities to the associated dimension (optional)
dtypes                  dict       Mapping of non-float entities to their data type (optional)
======================  =========  ================================================================================

In the `dimensions` dictionary, the keys are entities as given in the tabular data in
the entity column. The values are lists of column names as used in the tabular data,
i.e. including the terminology.
To avoid repeating dimension information for many entities with the same dimensions,
it is possible to use `*` as the entity name in the dimensions dict, which will be used
as a default for all entities not explicitly listed.
Dimension information has to be given for all entities, i.e. if no default dimensions
are specified using `*`, there has to exist an entry in the dimensions dict for each
unique value in the entity column in the tabular data.

Data Format
-----------

Numeric values are given unquoted and string values are quoted with ``"``.
Missing information is denoted by an empty string ``""``.
