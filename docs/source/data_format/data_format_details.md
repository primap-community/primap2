(data_format_details)=
# Data format details

Data is stored in xarray `dataset` objects with specified dimensions, coordinates,
data variables, and attrs.

## Dimensions

For all datasets, the dimensions for the time and the area are required, and other
dimensions and coordinates can be given if necessary.
For all dimensions, defined names have to be used and additional metadata about the
dimensions is stored in the datasets `attrs`.
The dimensions are:

| dimension       | dimension key           | req. | notes                     | attrs                             |
|-----------------|-------------------------|------|---------------------------|-----------------------------------|
| `time`          | `time`                  | ✗    | for periods, the start    |                                   |
| `area`          | `area (<category-set>)` | ✗    | pre-defined category sets | `'area': 'area (<category set>)'` |
| `category`      | `category (<c-set>)`    |      | primary category          | `'cat': 'category (<c-set>)'`     |
| sec. categories | `<type> (<c-set>)`      |      | there can be multiple     |                                   |
| `scenario`      | `scenario (<c-set>)`    |      |                           | `'scen': 'scenario (<c-set>)'`    |
| `provenance`    | `provenance`            |      | values from fixed set     |                                   |
| `model`         | `model`                 |      |                           |                                   |
| `source`        | `source`                | ✗    | a short source identifier |                                   |

For some dimensions, the meaning of the data is directly visible from the data type
(`time` uses an xarray datetime data type) or the values come from a pre-defined list
(provenance).
For the other dimensions, values come from a set of categories (denoted `<category-set>`
or shorter `<c-set>` in the table), such as the ISO-3166-1 three-letter country
abbreviations denoting the area or IPCC 2006 categories being used as the primary
category.
In this case, the name of the used category-set according to
[climate-categories](https://climate-categories.readthedocs.io/en/latest/readme.html#included-categorizations)
is included directly in the dimension key in brackets, and a translation from a generic
name to the dimension key is included in the `attrs`.

Most commonly, data have either no category (for example, population data) or one
primary category (for example, most CO2 emissions data).
Therefore, the primary category is not required, and if it is used, it is
denoted as the `category`.
However, some data sets have more than one categorization, for example the FAO land-use
emissions data, that are categorized according to agricultural sector and animal type.
Therefore, it is possible to include arbitrary secondary categories, where the
dimension key is then formed from the dimension or type (`<type>` in the table) and the
category-set (for example, `animal (FAOSTAT)`).

Additional rules:

- The valid values for the `provenance` are `measured`, `projected`, and
  `derived`.

## Additional Coordinates

Besides the coordinates defining dimensions, additional coordinates can be given, for
example to supply category names for the categories. Additional coordinates are not
required to have unique values.
The name of additional coordinates is not allowed to contain spaces, replace them
preferably with `_`.

## Data Variables

Each data variable has a name (key) and attributes (attrs).
The attributes are:

| attribute   | content                                                        | required          |
|-------------|----------------------------------------------------------------|-------------------|
| entity      | entity code, possibly from the dataset's entity category set   | yes               |
| gwp_context | global warming potential context used for calculating the data | if a gwp was used |
| units       | in which units the data is given                               | see rules below   |

For the `entity`, a category set (i.e. a terminology) should be defined for the
whole dataset in the dataset attributes using the key `entity_terminology` (see
below).
If the `entity_terminology` is defined, all entities in the dataset must be defined
in the terminology so that the exact meaning of entity codes is known.
If the `entity_terminology` is not defined, the meaning of the entities is not clearly
defined.

The name of the data variable (its key) is formed from the `entity` and the
`gwp_context`, if applicable.
If there is no `gwp_context`, the name is the entity.
If there is a `gwp_context`, the name is the entity, followed by the `gwp_context`
in parentheses, separated from the entity by a space.

Each single gas is contained in a data set only once, either as a mass or as a global
warming potential, and in only one global warming potential metric.
Gas baskets, whose entity is not a single gas known to
[openscm-units](https://openscm-units.readthedocs.io), are exempt: their composition is
unknown, so they can not be converted between global warming potentials and are
commonly given in several of them.

Units are required for all data variables with a `dtype` of `float`, while
for data with other data types, the units are required only where they make sense.
For example, data with an integer data type representing (human or animal) population
data requires units, while a data variable with a categorical data type representing
the evaluation method for each data point does not require units.
If the units are required, they can either be given by quantifying the data variable
using `pint_xarray`, or can be included in the variable attributes using the key
`units` as a string.
For storage, the dataset should not be quantified and the units should be given in the
`attrs`, but for calculations the dataset should be quantified using pint_xarray.
If given in the `attrs` as a string, the units must be parsable by
[openscm-units](https://openscm-units.readthedocs.io).

(data_format_details_dataset_attributes)=
## Dataset Attributes

Metadata about the dimensions and the data set as a whole is stored in the dataset
`attrs`.
The metadata about the dimensions is described above in the paragraph concerning
dimensions.
The other attributes with metadata about the dataset as a whole are:

| attribute          | description                                       | data type         |
|--------------------|---------------------------------------------------|-------------------|
| references         | citable references describing the data (e.g. DOI) | ``str``           |
| rights             | license or other usage restrictions               | ``str``           |
| contact            | who can answer questions about the data           | ``str`` (email)   |
| title              | a succinct description                            | ``str``           |
| comment            | longer form description                           | ``str``           |
| institution        | where the data originates                         | ``str``           |
| entity_terminology | terminology for data variable entities            | ``str``           |
| publication_date   | date of publication of the dataset                | ``datetime.date`` |

All of these attributes are optional.
If the `references` field starts with `doi:`, it is a doi, otherwise it is a
free-form literature reference.

These attributes describing the data set contents are inspired by the
[CF conventions](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents)
for the description of file contents.

The `entity_terminology` (if present) defines the meaning of the codes used in the
data variables' names and `entity` attributes.
In `entity_terminology`, the name of the used terminology is stored and the
terminology is defined elsewhere.

## Processing Information

How each data point was derived can be recorded as processing information, which is
stored in special data variables, one for each normal data variable it describes.
The name of these variables is `Processing of {var}` where `{var}` is the
full name of the data variable which is described. Additionally, the following
attributes are defined:

| attribute          | content                                           | required |
|--------------------|---------------------------------------------------|----------|
| entity             | The same as the name, i.e. `Processing of {var}`  | yes      |
| described_variable | The name of the described variable, i.e. `{var}`. | yes      |

The `gwp_context` and `units` attributes must not be given.

The array has the same dimensions as the data variable which is described, including
`time`.
It contains processing information as rich metadata types.
Therefore, the data type is `object` and it contains `primap2.ProcessingStep` objects,
the nodes of the processing history. A data point has a node if and only if it has data,
and the time of the node is the time of the data point. Data points without data are
marked by a null value (`None` or NaN), which is read back as `None`.
If a dataset contains processing information, every data variable needs processing
information.

Each `ProcessingStep` contains the following information:

| attribute   | type                                | description                                                                  |
|-------------|-------------------------------------|------------------------------------------------------------------------------|
| description | `ProcessingStepDescription`         | What the step did, shared by the nodes of a timeseries which the step set    |
| time        | `numpy.datetime64`                  | The time point of the data point which the step set                          |
| parents     | tuple of `ProcessingStep`           | The steps which set the values this value was computed from, empty for a step which created the value |

The parents can be at other time points than the node, e.g. the data points which were
used for a least-squares fit.
A `ProcessingStepDescription` contains the following information:

| attribute   | type          | description                                                                                 |
|-------------|---------------|---------------------------------------------------------------------------------------------|
| function    | str           | Name of the function which did the operation                                                |
| description | str           | Long-form description of the operation which was performed                                  |
| source      | str or `None` | If applicable, identifier for untracked data which entered in the operation               |
| variable    | str or `None` | The name of the data variable of the processed timeseries                                   |
| coords      | dict or `None`| The coordinates of the processed timeseries at the time of the operation, apart from "time" |

Descriptions are compared by identity, equal fields do not make descriptions equal.

Operations which combine several data points, like sums, have several parents, so the
processing history of a data point is a directed acyclic graph rather than a list.
Nodes are shared: the node of a data point can be a parent of the nodes of several
other data points, also in other variables.
`ProcessingStep.history()` returns all steps of the history in an order in
which every step comes after its parents, and
`ProcessingStep.format_history()` shows them as a tree, starting with the
last step. `primap2.format_timeseries_history` shows the histories of all data points
of a timeseries together.

### Storage of processing information in netCDF files

netCDF files can't hold Python objects, so `to_netcdf` writes the processing information
as tables of integers and strings, which `open_dataset` converts back.
The global attribute `processing_info_format` holds the version of the format, currently
`2`. The tables are one-dimensional variables with their own dimension, and only the
steps which are reachable from a data point are stored.

* `Processing of {var}` has the same dimensions as `{var}` and holds, for each data
  point, the index of its node in the table of nodes, or -1 if the data point has no
  node.
  The indices are `int32` (`int64` for more than 2**31-1 nodes), as are the integer
  columns of the tables below, which `to_netcdf` always compresses.
* Table of descriptions, dimension `processing step`, one entry per
  `ProcessingStepDescription`:

  | variable                           | content                                                       |
  |------------------------------------|---------------------------------------------------------------|
  | `processing step function`         | `function`                                                    |
  | `processing step description`      | `description`                                                 |
  | `processing step source`           | `source`, empty for `None`                                    |
  | `processing step source is null`   | 1 if `source` is `None`, else 0                               |
  | `processing step variable`         | `variable`, empty for `None`                                  |
  | `processing step variable is null` | 1 if `variable` is `None`, else 0                             |
  | `processing step coords`           | `coords` as a JSON object                                     |

* Table of nodes, dimension `processing node`, one entry per `ProcessingStep`. A node
  comes after its parents.

  | variable                 | content                                                         |
  |--------------------------|-----------------------------------------------------------------|
  | `processing node step`   | index of the node's description in the table of descriptions    |
  | `processing node time`   | `time`                                                          |
  | `processing node parents`| number of parents                                               |

* Table of parents, dimension `processing parent`: `processing parent node` holds the
  indices of the parents in the table of nodes. The parents of all nodes are stored one
  after the other in the order of the nodes, as many for each node as given by its
  `processing node parents`.
