import csv
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import strictyaml as sy
import xarray as xr
from loguru import logger
from ruamel.yaml import YAML

# entity is mandatory in the interchange format because it is transformed
# into the variables
# unit is mandatory in the interchange format because it is transformed
# into the pint units
INTERCHANGE_FORMAT_MANDATORY_COLUMNS = ["area", "source", "entity", "unit"]
INTERCHANGE_FORMAT_OPTIONAL_COLUMNS = ["category", "scenario", "provenance", "model"]
INTERCHANGE_FORMAT_COLUMN_ORDER = [
    "source",
    "scenario",
    "provenance",
    "model",
    "area",
    "entity",
    "unit",
    "category",
    "orig_cat_name",
    "cat_name_translation",
]

INTERCHANGE_FORMAT_STRICTYAML_SCHEMA = sy.Map(
    {
        sy.Optional("data_file"): sy.Str(),
        "attrs": sy.Map(
            {
                "area": sy.Str(),
                sy.Optional("cat"): sy.Str(),
                sy.Optional("sec_cats"): sy.Seq(sy.Str()),
                sy.Optional("scen"): sy.Str(),
                sy.Optional("references"): sy.Str(),
                sy.Optional("rights"): sy.Str(),
                sy.Optional("contact"): sy.Str(),
                sy.Optional("title"): sy.Str(),
                sy.Optional("comment"): sy.Str(),
                sy.Optional("institution"): sy.Str(),
                sy.Optional("entity_terminology"): sy.Str(),
                sy.Optional("publication_date"): sy.Datetime(),
            }
        ),
        "dimensions": sy.MapPattern(sy.Str(), sy.Seq(sy.Str())),
        "time_format": sy.Str(),
        sy.Optional("additional_coordinates"): sy.MapPattern(sy.Str(), sy.Str()),
        sy.Optional("dtypes"): sy.MapPattern(sy.Str(), sy.Str()),
    }
)


def dates_to_dimension(ds: xr.Dataset, time_format: str = "%Y") -> xr.DataArray:
    """
    This function converts a xr.Dataset where each time point is an individual
    data variable to a xr.DataArray with a time dimension.
    All variables which are not in the index are assumed to be time points.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with individual time points as data variables

    time_format : str
        format string for the time points. Default: %Y (for year only)

    Returns
    -------
    reduced: xr.DataArray
        xr.DataArray with the time as a dimension and time points as values
    """
    empty_vars = (x for x in ds if ds[x].count() == 0)
    da = ds.drop_vars(empty_vars).to_array("time")
    da["time"] = pd.to_datetime(da["time"].values, format=time_format, exact=False)
    return da


def metadata_for_variable(unit: str, variable: str) -> dict[str, str]:
    """Convert a primap2 unit and variable key to a metadata dict.

    Derives the information needed for the data variable's attrs dict from the unit
    and the variable's key.
    Takes GWP information from the variable name in primap2 style (if present).

    Parameters
    ----------
    unit: str
        unit to be stored in the attrs dict
    variable: str
        entity to process

    Returns
    -------
    attrs: dict
        attrs for use in DataArray.attrs

    Examples
    --------

    >>> metadata_for_variable("Gg CO2 / year", "Kyoto-GHG (SARGWP100)")
    {'units': 'Gg CO2 / year', 'gwp_context': 'SARGWP100', 'entity': 'Kyoto-GHG'}

    """
    attrs = {}

    if not isinstance(unit, str):
        unit = ""

    if unit != "no unit":
        attrs["units"] = unit

    regex_gwp = r"\s\(([A-Za-z0-9]*)\)$"
    gwp = re.findall(regex_gwp, variable)

    if gwp:
        attrs["gwp_context"] = gwp[0]

        regex_variable = r"^(.*)\s\([a-zA-z0-9]*\)$"
        entity = re.findall(regex_variable, variable)
        if not entity:
            logger.error(f"Can't extract entity from {variable}")
            raise ValueError(f"Can't extract entity from {variable}")
        attrs["entity"] = entity[0]
    else:
        attrs["entity"] = variable
    return attrs


def write_interchange_format(
    filepath: str | Path, data: pd.DataFrame, attrs: dict | None = None
) -> None:
    """Write dataset in interchange format to disk.

    Writes an interchange format dataset consisting of a pandas Dataframe and an
    additional meta data dict to disk. The data is stored in a csv file while the
    additional metadata is written to a yaml file.

    Parameters
    ----------
    filepath: str or pathlib.Path
        path and filename stem for the dataset. If a file ending is given it will be
        ignored and replaced by .csv for the data and .yaml for the metadata

    data: pandas.DataFrame
        DataFrame in PRIMAP2 interchange format

    attrs: dict, optional
        Interchange format meta data dict. Default: use data.attrs .
    """
    if attrs is None:
        attrs = data.attrs.copy()
    else:
        attrs = attrs.copy()

    # make sure filepath is a Path object
    filepath = Path(filepath)
    data_file = filepath.parent / (filepath.stem + ".csv")
    meta_file = filepath.parent / (filepath.stem + ".yaml")

    # write the data
    data.to_csv(data_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

    attrs["data_file"] = data_file.name

    yaml = YAML()
    # settings for strictyaml compatibility: don't use flow style or aliases
    yaml.default_flow_style = False
    yaml.representer.ignore_aliases = lambda x: True
    with meta_file.open("w") as fd:
        yaml.dump(attrs, fd)


def read_interchange_format(
    filepath: str | Path,
) -> pd.DataFrame:
    """Read a dataset in the interchange format from disk into memory.

    Reads an interchange format dataset from disk. The data is stored
    in a csv file while the additional metadata is stored in a yaml
    file. This function takes the yaml file as parameter, the data file is specified
    in the yaml file. If no or a wrong ending is given the function tries to load
    a file by the same name with the ending `.yaml`.

    Parameters
    ----------
    filepath: str or pathlib.Path
        path and filename for the dataset (the yaml file, not data file).

    Returns
    -------
    data: pandas.DataFrame
        DataFrame with the read data in PRIMAP2 interchange format
    """
    filepath = Path(filepath)
    if not filepath.exists():
        filepath = filepath.with_suffix(".yaml")
    with filepath.open() as meta_file:
        yaml = sy.load(meta_file.read(), schema=INTERCHANGE_FORMAT_STRICTYAML_SCHEMA)
        meta_data = yaml.data

    data_file = filepath.parent / meta_data.get("data_file", filepath.stem + ".csv")
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found at {data_file}.")

    data = pd.read_csv(data_file, dtype=object)
    data.attrs = meta_data

    # strictyaml parses a datetime, we only want a date
    if "publication_date" in data.attrs["attrs"]:
        data.attrs["attrs"]["publication_date"] = data.attrs["attrs"][
            "publication_date"
        ].date()
    # already read in
    if "data_file" in data.attrs:
        del data.attrs["data_file"]

    return data


def from_interchange_format(
    data: pd.DataFrame,
    attrs: dict | None = None,
    max_array_size: int = 1024 * 1024 * 1024,
) -> xr.Dataset:
    """Convert dataset from the interchange format to the standard PRIMAP2 format.

    Converts an interchange format DataFrame with added metadata to a PRIMAP2 xarray
    data structure. All column names and attrs are expected to be already in PRIMAP2
    format as defined for the interchange format. The attrs dict is given explicitly as
    the attrs functionality in pandas is experimental.

    Parameters
    ----------
    data: pd.DataFrame
        pandas DataFrame in PRIMAP2 interchange format.
    attrs: dict, optional
        attrs dict as defined for the PRIMAP2 interchange format. Default: use
        data.attrs.
    max_array_size: int, optional
        Maximum permitted projected array size. Larger sizes will raise an exception.
        Default: 1 G, corresponding to about 4 GB of memory usage.

    Returns
    -------
    obj: xr.Dataset
        xr dataset with the converted data
    """
    if attrs is None:
        attrs = data.attrs.copy()

    if "entity_terminology" in attrs["attrs"]:
        entity_col = f"entity ({attrs['attrs']['entity_terminology']})"
    else:
        entity_col = "entity"

    if "additional_coordinates" not in attrs:
        attrs["additional_coordinates"] = {}

    # build dicts for additional coordinates
    add_coord_mapping_dicts = {}
    for coord in attrs["additional_coordinates"].keys():
        values = data[[coord, attrs["additional_coordinates"][coord]]]
        values = values.drop_duplicates()
        dim_values = list(values[attrs["additional_coordinates"][coord]])
        coord_values = list(values[coord])
        if len(coord_values) != len(set(dim_values)):
            logger.error(
                f"Different secondary coordinate values for given first coordinate "
                f"value for {coord}."
            )
            raise ValueError(
                f"Different secondary coordinate values for given first coordinate "
                f"value for {coord}."
            )

        add_coord_mapping_dicts[coord] = dict(
            zip(dim_values, coord_values, strict=False)
        )

    # drop additional coordinates. make a copy first to not alter input DF
    data_drop = data.drop(columns=attrs["additional_coordinates"].keys(), inplace=False)

    # find the time columns
    if_index_cols = set(itertools.chain(*attrs["dimensions"].values()))
    time_cols = set(data_drop.columns.values) - if_index_cols

    # convert to xarray
    data_xr = data_drop.to_xarray()
    index_cols = if_index_cols - {"unit", "time"}
    data_xr = data_xr.set_index({"index": list(index_cols)})
    # take the units out as they increase dimensionality and we have only one unit per
    # entity/variable
    units = data_xr["unit"]
    del data_xr["unit"]

    # build full dimensions dict from specification with default from entry "*"
    entities = np.unique(data_xr[entity_col].values)
    dimensions = attrs["dimensions"]
    for entity in entities:
        if entity not in dimensions:
            dimensions[entity] = dimensions["*"]
    if "*" in dimensions:
        del dimensions["*"]

    # build full dtypes dict from specification with default float
    dtypes = attrs.get("dtypes", {})
    for entity in entities:
        if entity not in dtypes:
            dtypes[entity] = "float"

    # check resulting shape to estimate memory consumption
    dim_lens = {dim: len(np.unique(data_xr[dim].dropna("index"))) for dim in index_cols}
    dim_lens["time"] = len(time_cols)
    shapes = []
    for _, dims in dimensions.items():
        shapes.append([dim_lens[dim] for dim in dims if dim != "unit"])
    array_size = sum(np.prod(shape) for shape in shapes)
    logger.debug(f"Expected array shapes: {shapes}, resulting in size {array_size:,}.")
    if array_size > max_array_size:
        logger.error(
            f"Set with {len(shapes)} entities and a total of {len(index_cols)} "
            f"dimensions will have a size of {array_size:,} "
            f"due to the shapes {shapes}. Aborting to avoid out-of-memory errors. To "
            f"continue, raise max_array_size (currently {max_array_size:,})."
        )
        raise ValueError(
            f"Resulting array too large: {array_size:,} > {max_array_size:,}. To "
            f"continue, raise max_array_size."
        )

    # convert time to dimension and entity to variables.
    if "time_format" in attrs:
        da = dates_to_dimension(data_xr, time_format=attrs["time_format"])
    else:
        da = dates_to_dimension(data_xr)
    data_vars = {}
    for entity, dims in dimensions.items():
        da_entity = da.loc[{entity_col: entity}]
        # we still have a full MultiIndex, so trim it to the relevant dimensions
        da_entity = da_entity.reset_index(list(index_cols - set(dims)), drop=True)
        # depending on the version of xarray, an atomic, 0-dimensional coord
        # for the entity remains. we have to remove it to be able to combine the
        # dataset afterwards.
        if entity_col in da_entity.coords:
            da_entity = da_entity.drop_vars(entity_col)
        # now we can safely unstack the index
        data_vars[entity] = da_entity.unstack("index").astype(dtypes[entity])

    data_xr = xr.Dataset(data_vars)

    # add the additional coordinates
    for coord in attrs["additional_coordinates"].keys():
        dim_values_xr = list(data_xr[attrs["additional_coordinates"][coord]].values)
        coord_values_ordered = [
            add_coord_mapping_dicts[coord][value] for value in dim_values_xr
        ]
        data_xr = data_xr.assign_coords(
            {
                coord: xr.DataArray(
                    data=np.array(coord_values_ordered),
                    coords={
                        attrs["additional_coordinates"][coord]: data_xr.coords[
                            attrs["additional_coordinates"][coord]
                        ]
                    },
                    dims=[attrs["additional_coordinates"][coord]],
                )
            }
        )

    # fill the entity/variable attributes
    for variable in data_xr:
        csv_units = np.unique(units.loc[{entity_col: variable}], equal_nan=True)
        if len(csv_units) > 1 and any(isinstance(x, str) for x in csv_units):
            logger.error(
                f"More than one unit for entity {variable!r}: {csv_units!r}. "
                + "There is an error in the unit harmonization."
            )
            raise ValueError(f"More than one unit for {variable!r}: {csv_units!r}.")
        data_xr[variable].attrs = metadata_for_variable(csv_units[0], variable)

    # add the dataset wide attributes
    data_xr.attrs = attrs["attrs"]

    data_xr = data_xr.pr.quantify()

    data_xr.pr.ensure_valid()
    return data_xr
