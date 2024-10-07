.. _data_format_details:

===================
Data format details
===================

Data is stored in xarray ``dataset`` objects with specified dimensions, coordinates,
data variables, and attrs.

Dimensions
----------

For all datasets, the dimensions for the time and the area are required, and other
dimensions and coordinates can be given if necessary.
For all dimensions, defined names have to be used and additional metadata about the
dimensions is stored in the datasets ``attrs``.
The dimensions are:

===============  =====================  ========  ======================================  =======================================
dimension        dimension key          required  notes                                   attrs
---------------  ---------------------  --------  --------------------------------------  ---------------------------------------
time             time                   ✗         for periods, the *start* of the period
area             area (<category-set>)  ✗         must be a pre-defined category set      ``'area': 'area (<category set>)'``
category         category (<c-set>)               primary category                        ``'cat': 'category (<c-set>)'``
sec. categories  <type> (<c-set>)                 there can be multiple                   ``'sec_cats': ['<type> (<c-set>)', …]``
scenario         scenario (<c-set>)                                                       ``'scen': 'scenario (<c-set>)'``
provenance       provenance                       values from fixed set
model            model                            model should be from a predefined list
source           source                 ✗         a short source identifier
===============  =====================  ========  ======================================  =======================================

For some dimensions, the meaning of the data is directly visible from the data type
(``time`` uses an xarray datetime data type) or the values come from a pre-defined list
(provenance, model).
For the other dimensions, values come from a set of categories (denoted <category-set>
or shorter <c-set> in the table), such as the ISO-3166-1 three-letter country
abbreviations denoting the area or IPCC 2006 categories being used as the primary
category.
In this case, the used category-set is included directly in the dimension key in
brackets, and a translation from a generic name to the dimension key is included in the
``attrs``.

Most commonly, data have either no category (for example, population data) or one
primary category (for example, most CO2 emissions data).
Therefore, the primary category is not required, and if it is used, it is
denoted as the ``category``.
However, some data sets have more than one categorization, for example the FAO land-use
emissions data, that are categorized according to agricultural sector and animal type.
Therefore, it is possible to include arbitrary secondary categories, where the
dimension key is then formed from the dimension or type (<type> in the table) and the
category-set (for example, ``animal (FAOSTAT)``).

Additional rules:

* The valid values for the ``provenance`` are ``measured``, ``projected``, and
  ``derived``.

Additional Coordinates
----------------------

Besides the coordinates defining dimensions, additional coordinates can be given, for
example to supply category names for the categories. Additional coordinates are not
required to have unique values.
The name of additional coordinates is not allowed to contain spaces, replace them
preferably with ``_``.

Data Variables
--------------

Each data variable has a name (key) and attributes (attrs).
The attributes are:

===========  ========================================================================  ============================
attribute    content                                                                   required
-----------  ------------------------------------------------------------------------  ----------------------------
entity       entity code, possibly from the dataset's entity category set              yes
gwp_context  which global warming potential context was used for calculating the data  if a gwp was used
units        in which units the data is given                                          see rules below
===========  ========================================================================  ============================

For the ``entity``, a category set (i.e. a terminology) should be defined for the
whole dataset in the dataset attributes using the key ``entity_terminology`` (see
below).
If the ``entity_terminology`` is defined, all entities in the dataset must be defined
in the terminology so that the exact meaning of entity codes is known.
If the ``entity_terminology`` is not defined, the meaning of the entities is not clearly
defined.

The name of the data variable (its key) is formed from the ``entity`` and the
``gwp_context``, if applicable.
If there is no ``gwp_context``, the name is the entity.
If there is a ``gwp_context``, the name is the entity, followed by the ``gwp_context``
in parentheses, separated from the entity by a space.

Units are required for all data variables with a ``dtype`` of ``float``, while
for data with other data types, the units are required only where they make sense.
For example, data with an integer data type representing (human or animal) population
data requires units, while a data variable with a categorical data type representing
the evaluation method for each data point does not require units.
If the units are required, they can either be given by quantifying the data variable
using ``pint_xarray``, or can be included in the variable attributes using the key
``units`` as a string.
For storage, the dataset should not be quantified and the units should be given in the
``attrs``, but for calculations the dataset should be quantified using pint_xarray.
If given in the ``attrs`` as a string, the units must be parsable by
`openscm-units <https://openscm-units.readthedocs.io>`_.

Dataset Attributes
------------------

Metadata about the dimensions and the data set as a whole is stored in the dataset
``attrs``.
The metadata about the dimensions is described above in the paragraph concerning
dimensions.
The other attributes with metadata about the dataset as a whole are:

==================  ========================================  =========================================
attribute           description                               data type
------------------  ----------------------------------------  -----------------------------------------
references          citable reference(s) describing the data  free-form ``str`` (ideally URL)
rights              license or other usage restrictions       free-form ``str``
contact             who can answer questions about the data   free-form ``str`` (usually email)
title               a succinct description                    free-form ``str``
comment             longer form description                   free-form ``str``
institution         where the data originates                 free-form ``str``
entity_terminology  terminology for data variable entities    ``str``
publication_date    date of publication of the dataset        ``datetime.date``
==================  ========================================  =========================================

All of these attributes are optional.
If the ``references`` field starts with ``doi:``, it is a doi, otherwise it is a
free-form literature reference.

These attributes describing the data set contents are inspired by the
`CF conventions <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_
for the description of file contents.

The ``entity_terminology`` (if present) defines the meaning of the codes used in the
data variables' names and ``entity`` attributes.
In ``entity_terminology``, the name of the used terminology is stored and the
terminology is defined elsewhere.

Timeseries-Level Processing Information
---------------------------------------

To record processing information, rich processing information can be stored in
special data variables.
These store processing information for a specific normal data variable.
For each timeseries in the normal data variable, a list of processing steps is kept.
The name of these variables is ``Processing of {var}`` where ``{var}`` is the
full name of the data variable which is described. Additionally, the following
attributes are defined:

==================  ========================================================================  ============================
attribute           content                                                                   required
------------------  ------------------------------------------------------------------------  ----------------------------
entity              The same as the name, i.e. ``Processing of {var}``                        yes
described_variable  The name of the described variable, i.e. ``{var}``.                       yes
==================  ========================================================================  ============================

The ``gwp_context`` and ``units`` attributes must not be given.

``time`` must not be a dimension of the array, all other dimensions must be the same
as the data variable which is described.
The data array contains processing information as rich metadata types.
Therefore, the data type is ``object`` and it contains
``primap2.TimeseriesProcessingDescription`` objects.
Each ``TimeseriesProcessingDescription`` object comprises multiple
``primap2.ProcessingStepDescription`` objects.
Each ``ProcessingStepDescription`` contains the following information:

===========  =================================  =============================================================================
attribute    type                               description
-----------  ---------------------------------  -----------------------------------------------------------------------------
time         ``np.datetime64`` or string "all"  The time points affected by the operation, or "all" to denote operations on the whole timeseries
function     str                                Identifier for the function which did the operation
description  str                                Long-form description of the operation which was performed
source       str or ``None``                    If applicable, identifier for the data source which was used in the operation
===========  =================================  =============================================================================
