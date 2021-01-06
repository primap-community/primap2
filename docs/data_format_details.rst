===================
Data format details
===================

Coordinates
-----------

For all datasets, the coordinates for the time and the area are required, and other
coordinates can be given if necessary.
For all coordinates, defined names have to be used and additional metadata about the
coordinates is stored in the datasets ``attrs``.
The coordinates are:

===============  =====================  ========  ===========================================  ===================================
coordinate       coordinate key         required  notes                                        attrs
---------------  ---------------------  --------  -------------------------------------------  -----------------------------------
time             time                   ✗         for periods, the *start* of the period
area             area (<category-set>)  ✗         must be a pre-defined category set           'area': 'area (<category set>)'
category         category (<c-set>)               primary category                             'cat': 'category (<c-set>)'
sec. categories  <type> (<c-set>)                 there can be multiple                        'sec_cats': ['<type> (<c-set>)', …]
scenario         scenario (<c-set>)                                                            'scen': 'scenario (<c-set>)'
provenance       provenance                       values are measured, projected, and derived
model            model                            model should be from a predefined list
source           source                 ✗         a short source identifier
===============  =====================  ========  ===========================================  ===================================

For some coordinates, the meaning of the data is directly visible from the data type
(time uses an xarray datetime data type) or the values come from a pre-defined list
(provenance, model).
For the other coordinates, values come from a set of categories (denoted <category-set>
or shorter <c-set> in the table), such as the ISO-3166-1 three-letter country
abbreviations denoting the area or IPCC 2006 categories being used as the primary
category.
In this case, the used category-set is included directly in the coordinate key in
brackets, and a translation from a generic name to the coordinate key is included in the
``attrs``.

Most commonly, data have either no category (for example, population data) or one
primary category (for example, most CO2 emissions data).
Therefore, the primary category is not required, and if it is used, it is specifically
denoted as the ``category``.
However, some data sets have more than one categorization, for example the FAO land-use
emissions data, that are categorized according to agricultural sector and animal type.
Therefore, it is possible to include arbitrary secondary categories, where the
coordinate key is then formed from the dimension or type (<type> in the table) and the
category-set (for example, ``animal (FAOSTAT)``)
