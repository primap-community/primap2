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

===============  =====================  ========  ========================================================  ===================================
dimension        dimension key          required  notes                                                     attrs
---------------  ---------------------  --------  --------------------------------------------------------  -----------------------------------
time             time                   ✗         for periods, the *start* of the period
area             area (<category-set>)  ✗         must be a pre-defined category set                        'area': 'area (<category set>)'
category         category (<c-set>)               primary category                                          'cat': 'category (<c-set>)'
sec. categories  <type> (<c-set>)                 there can be multiple                                     'sec_cats': ['<type> (<c-set>)', …]
scenario         scenario (<c-set>)                                                                         'scen': 'scenario (<c-set>)'
provenance       provenance                       values are ``measured``, ``projected``, and ``derived``
model            model                            model should be from a predefined list
source           source                 ✗         a short source identifier
===============  =====================  ========  ========================================================  ===================================

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

* The size of the ``source`` dimension must be exactly 1.

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
entity       what gas emission or other entity is measured                             yes
gwp_context  which global warming potential context was used for calculating the data  if a gwp was used
units        in which units the data is given                                          if dataset is not quantified
===========  ========================================================================  ============================

Additional rules:

* If the entity is a gas emission rate (e.g. mass of CO2 emitted per year), only the
  gas is used as the entity.
* If the entity is the human population, only ``population`` is used as the entity.
* Gases, gwp_contexts and units use the names used in
  `openscm-units <https://openscm-units.readthedocs.io/en/latest/>`_.
* The unit must be given in the attributes if the dataset is not quantified
  using pint_xarray.
  For storage, the dataset should not be quantified, but for calculations the dataset
  should be quantified using pint_xarray.

The name of the data variable should be the entity, and if it has a gwp_context, the
gwp_context in parentheses, separated from the entity by a space.

Dataset Attributes
------------------

Metadata about the dimensions and the data set as a whole is stored in the dataset
``attrs``.
The metadata about the dimensions is described above in the paragraph concerning
dimensions.
The other attributes with metadata about the dataset as a whole are:

===========  ========================================  ============================================
attribute    description                               data type
-----------  ----------------------------------------  --------------------------------------------
references   citable reference(s) describing the data  ``str`` starting with ``doi:`` or free-form
rights       license or other usage restrictions       free-form ``str``
contact      who can answer questions about the data   usually email address, but free-form ``str``
title        a succinct description                    free-form ``str``
comment      longer form description                   free-form ``str``
institution  where the data originates                 free-form ``str``
history      processing steps done on the data         ``str`` with specific rules (see text)
===========  ========================================  ============================================

All of these attributes are optional.
If the ``references`` field starts with ``doi:``, it is a doi, otherwise it is a
free-form literature reference.
In the ``history`` field, an audit trail of modifications can be stored. Steps are
separated by a newline character, and processing steps should append to the field.

These attributes describing the data set contents are inspired by the
`CF conventions <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_
for the description of file contents.
