import itertools
import re
from pathlib import Path
from typing import IO, Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from .._units import ureg

SEC_CATS_PREFIX = "sec_cats__"
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
]


def convert_dataframe_units_primap_to_primap2(
    data_frame: pd.DataFrame, unit_col: str = "unit", entity_col: str = "entity"
):
    """
    This function converts the emissions module style units which usually neither carry
    information about the substance nor about the time to primap2 units. The function
    also handles the exception cases where PRIMAP units do contain information about the
    substance (e.g. GtC).

    The function operates in place.

    Parameters
    ----------

    data_frame : pandas.Dataframe
        data_frame with IPCC category codes in in PRIMAP format (IPCC1996 and IPCC2006
        can be converted)
    unit_col : str
        name of the column that contains the unit. Default: "unit"
    entity_col : str
        name of the column that contains the entity. Default: "entity"

    Returns
    -------
    no return value, data_frame is altered in place

    """
    # add variable_unit column with replaced units.
    # special units will be replaced later
    data_frame["entity_unit"] = (
        data_frame[unit_col].astype(str) + "<<>>" + data_frame[entity_col].astype(str)
    )

    # get unique entity_unit combinations
    unique_entity_unit = data_frame["entity_unit"].unique()
    for entity_unit in unique_entity_unit:
        [cur_entity, cur_unit] = re.split("<<>>", entity_unit)
        data_frame.loc[
            data_frame["entity_unit"] == entity_unit, "entity_unit"
        ] = convert_unit_primap_to_primap2(cur_entity, cur_unit)

    data_frame[unit_col] = data_frame["entity_unit"]
    data_frame.drop(columns=["entity_unit"], inplace=True)
    return data_frame


def convert_unit_primap_to_primap2(unit: str, entity: str) -> str:
    """
    This function converts the emissions module style units which usually neither carry
    information about the substance nor about the time to primap2 units. The function
    also handles the exception cases where PRIMAP units do contain information about the
    substance (e.g. GtC).

    Parameters
    ----------

    unit : str
        unit to convert
    entity : str
        entity for which the conversion takes place

    Returns
    -------
    :str: converted unit

    """

    # check inputs
    if entity == "":
        logger.warning("Input entity is empty. Nothing converted.")
        return "error_" + unit + "_" + entity
    if unit == "":
        logger.warning("Input unit is empty. Nothing converted.")
        return "error_" + unit + "_" + entity

    # define exceptions
    exception_units = {
        "CO2eq": "CO2",  # convert to just CO2
        "<entity>N": "N",
        "C": "C",  # don't add variable here
    }

    # basic units
    basic_units = ["g", "t"]

    # prefixes
    si_unit_multipliers = ["", "k", "M", "G", "T", "P", "E", "Z", "Y"]

    # combines basic units with prefixes
    units_prefixes = list(itertools.product(si_unit_multipliers, basic_units))
    units_prefixes = [i[0] + i[1] for i in units_prefixes]

    # time information to add
    time_frame_str = " / yr"

    # build regexp to match the basic units with prefixes in units
    regexp_str = "("
    for this_unit in units_prefixes:
        regexp_str = regexp_str + this_unit + "|"
    regexp_str = regexp_str[0:-1] + ")"

    # add entity and time frame to unit
    # special units will be replaced later
    unit_entity = unit + " " + entity + time_frame_str

    # check if unit has prefix
    match_pref = re.search(regexp_str, unit_entity)
    if match_pref is None:
        logger.warning("No unit prefix matched for unit. " + unit_entity)
        return "error_" + unit + "_" + entity

    # check if exception unit
    is_ex_unit = [
        re.match(regexp_str + ex_unit.replace("<entity>", entity) + "$", unit)
        is not None
        for ex_unit in exception_units
    ]

    if any(is_ex_unit):
        # we have an exception unit
        exception_unit = list(exception_units.keys())[is_ex_unit.index(True)]
        # first get the prefix and basic unit (e.g. Gt)
        pref_basic = match_pref.group(0)
        # now build the replacement
        converted_unit = (
            pref_basic + " " + exception_units[exception_unit] + time_frame_str
        )
    else:
        # standard unit
        converted_unit = unit_entity

    return converted_unit


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
    da = ds.drop_vars(empty_vars).to_array("time").unstack()
    da["time"] = pd.to_datetime(da["time"].values, format=time_format, exact=False)
    return da


def code_invalid_warn(code: str, message: str) -> str:
    """Log a warning and return an error code."""
    logger.warning(
        f"Category code {code!r} does not conform to specifications: {message}"
    )
    return "error_" + code


def convert_ipcc_code_primap_to_primap2(code: str) -> str:
    """
    This function converts IPCC emissions category codes from the PRIMAP-format to
    the pyCPA format which is closer to the original (but without all the dots)

    Codes that are not part of the official hierarchy (starting with IPCM or CATM)
    are not converted but returned without the 'CAT' or 'IPC' prefix unless the
    conversion is explicitly defined in the code_mapping dict.
    TODO: add other primap terminologies?

    Parameters
    ----------

    code: str
        String containing the IPCC code in PRIMAP format (IPCC1996 and IPCC2006 can be
        converted). The PRIMAP format codes consist of upper case letters and numbers
        only. These are converted back to the original formatting which includes lower
        case letters and roman numerals.

    Returns
    -------

    :str:
        string containing the category code in primap2 format

    Examples
    --------

    >>> convert_ipcc_code_primap_to_primap2("IPC1A3B34")
    '1.A.3.b.iii.4'

    """

    arabic_to_roman = {
        "1": "i",
        "2": "ii",
        "3": "iii",
        "4": "iv",
        "5": "v",
        "6": "vi",
        "7": "vii",
        "8": "viii",
        "9": "ix",
    }

    code_mapping = {
        "MAG": "M.AG",
        "MAGELV": "M.AG.ELV",
        "MBK": "M.BK",
        "MBKA": "M.BK.A",
        "MBKM": "M.BK.M",
        "M1A2M": "M.1.A.2.m",
        "MLULUCF": "M.LULUCF",
        "MMULTIOP": "M.MULTIOP",
        "M0EL": "M.0.EL",
    }

    if len(code) < 4:
        return code_invalid_warn(code, "Too short to be a PRIMAP IPCC code.")
    if code[0:3] not in ["IPC", "CAT"]:
        return code_invalid_warn(
            code, "Prefix is missing, must be one of 'IPC' or 'CAT'."
        )

    # it's an IPCC code. convert it
    # check if it's a custom code (beginning with 'M'). Currently these are the same
    # in pyCPA as in PRIMAP
    if code[3] == "M":
        new_code = code[3:]
        if new_code in code_mapping.keys():
            new_code = code_mapping[new_code]

    else:
        # actual conversion happening here
        # only work with the part without 'IPC' or 'CAT'
        code_remaining = code[3:]

        # first level is a digit
        if code_remaining[0].isdigit():
            new_code = code_remaining[0]
        else:
            return code_invalid_warn(code, "No digit found on first level.")
        # second level is a letter
        if len(code_remaining) > 1:
            code_remaining = code_remaining[1:]
            if code_remaining[0].isalpha():
                new_code = new_code + "." + code_remaining[0]
            else:
                return code_invalid_warn(code, "No letter found on second level.")
            # third level is a number. might be more than one char, so use regexp
            if len(code_remaining) > 1:
                code_remaining = code_remaining[1:]
                match = re.match("^[0-9]+", code_remaining)
                if match is not None:
                    new_code = new_code + "." + match.group(0)
                else:
                    return code_invalid_warn(code, "No number found on third level.")
                # fourth level is a letter. has to be transformed to lower case
                if len(code_remaining) > len(match.group(0)):
                    code_remaining = code_remaining[len(match.group(0)) :]
                    if code_remaining[0].isalpha():
                        new_code = new_code + "." + code_remaining[0].lower()
                    else:
                        return code_invalid_warn(
                            code, "No letter found on fourth level."
                        )
                    # fifth level is digit in PRIMAP1 format but roman numeral in IPCC
                    # and PRIMAP2
                    if len(code_remaining) > 1:
                        code_remaining = code_remaining[1:]
                        if code_remaining[0].isdigit():
                            new_code = (
                                new_code + "." + arabic_to_roman[code_remaining[0]]
                            )
                        else:
                            return code_invalid_warn(
                                code, "No digit found on fifth level."
                            )
                        # sixth and last level is a number.
                        if len(code_remaining) > 1:
                            code_remaining = code_remaining[1:]
                            match = re.match("^[0-9]+", code_remaining)
                            if match is not None:
                                new_code = new_code + "." + match.group(0)
                                # check if anything left
                                if not code_remaining == match.group(0):
                                    return code_invalid_warn(
                                        code, "Chars left after sixth level."
                                    )
                            else:
                                return code_invalid_warn(
                                    code, "No number found on sixth level."
                                )

    return new_code


def read_wide_csv_file_if(
    filepath_or_buffer: Union[str, Path, IO],
    *,
    coords_cols: Dict[str, str],
    coords_defaults: Optional[Dict[str, Any]] = None,
    coords_terminologies: Optional[Dict[str, str]] = None,
    coords_value_mapping: Optional[Dict[str, Any]] = None,
    filter_keep: Optional[Dict[str, Dict[str, Any]]] = None,
    filter_remove: Optional[Dict[str, Dict[str, Any]]] = None,
    meta_data: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    This function reads a csv file and returns the data as a pandas DataFrame in PRIMAP2
    interchange format.

    Columns can be renamed or filled with default vales to match the PRIMAP2 structure.
    Where we refer to "dimensions" in the parameter description below we mean the basic
    dimension names without the added terminology (e.g. "area" not "area (ISO3)"). The
    terminology information will be added by this function. You can not use the short
    dimension names in the attributes (e.g. "cat" instead of "category")

    Currently duplicate datapoints will not be detected.

    TODO: enable filtering through query strings

    TODO: sort the metadata columns before returning the data

    Parameters
    ----------

    filepath_or_buffer: str, pathlib.Path, or file-like
        Wide CSV file which will be read.

    coords_cols : dict
        Dict where the keys are column names in the files to be read and the value is
        the dimension in PRIMAP2. For secondary categories use a ``sec_cats__`` prefix.

    coords_defaults : dict, optional
        Dict for default values of coordinates / dimensions not given in the csv files.
        The keys are the dimension or metadata names and the values are the values for
        the dimensions or metadata. The distinction between dimension and metadata is
        done automatically on the basis of mandatory dimensions. For secondary
        categories use a ``sec_cats__`` prefix.

    coords_terminologies : dict, optional
        Dict defining the terminologies used for the different coordinates (e.g. ISO3
        for area). Only possible coordinates here are: area, category, scenario, and
        secondary categories. For secondary categories use a ``sec_cats__`` prefix.
        All entries different from "area", "category", "scenario", and "sec_cats__<name>
        will raise an error.

    coords_value_mapping : dict, optional
        A dict with primap2 dimension names as keys. Values are dicts with input values
        as keys and output values as values. A standard use case is to map gas names
        from input data to the standardized names used in primap2.
        Alternatively a value can also be a function which transforms one CSV metadata
        value into the new metadata value.
        A third possibility is to give a string as a value, which defines a rule for
        translating metadata values. The only defined rule at the moment is "PRIMAP1"
        which can be used for the "category", "entity", and "unit" columns to translate
        from PRIMAP1 metadata to PRIMAP2 metadata.
        Meta mapping is only possible for columns read from the input file not for
        columns defined via columns_defaults - for defaults, specify the mapped values
        directly in columns_defaults.

    filter_keep : dict, optional
        Dict defining filters of data to keep. Filtering is done before metadata
        mapping, so use original metadata values to define the filter. Column names are
        as in the csv file. Each entry in the dict defines an individual filter.
        The names of the filters have no relevance. Default: keep all data.

    filter_remove : dict, optional
        Dict defining filters of data to remove. Filtering is done before metadata
        mapping, so use original metadata values to define the filter. Column names are
        as in the csv file. Each entry in the dict defines an individual filter.
        The names of the filters have no relevance.

    meta_data : dict, optional
        Meta data for the whole dataset. Will end up in the dataset-wide attrs. Allowed
        keys are "references", "rights", "contact", "title", "comment", "institution",
        and "history". Documentation about the format and meaning of the meta data can
        be found in the
        `data format documentation <https://primap2.readthedocs.io/en/stable/data_format_details.html#dataset-attributes>`_.  # noqa: E501

    Returns
    -------

    obj: pd.DataFrame
        pandas DataFrame with the read data


    Examples
    --------

    *Example for meta_mapping*::

        meta_mapping = {
            'pyCPA_col_1': {'col_1_value_1_in': 'col_1_value_1_out',
                            'col_1_value_2_in': 'col_1_value_2_out',
                            },
            'pyCPA_col_2': {'col_2_value_1_in': 'col_2_value_1_out',
                            'col_2_value_2_in': 'col_2_value_2_out',
                            },
        }

    *Example for filter_keep*::

            filter_keep = {
                'f_1': {'variable': ['CO2', 'CH4'], 'region': 'USA'},
                'f_2': {'variable': 'N2O'}
            }

    This example filter keeps all CO2 and CH4 data for the USA and N2O data for all
    countries

    *Example for filter_remove*::

        filter_remove = {
            'f_1': {'scenario': 'HISTORY'},
        }

    This filter removes all data with 'HISTORY' as scenario

    """
    # Check and prepare arguments
    if coords_defaults is None:
        coords_defaults = {}
    if coords_terminologies is None:
        coords_terminologies = {}
    if meta_data is None:
        attrs = {}
    else:
        attrs = meta_data.copy()

    check_mandatory_dimensions(coords_cols, coords_defaults)
    check_overlapping_specifications(coords_cols, coords_defaults)

    data = read_csv(filepath_or_buffer, coords_cols)

    filter_data(data, filter_keep, filter_remove)

    add_dimensions_from_defaults(data, coords_defaults)

    if coords_value_mapping is not None:
        map_metadata(data, coords_cols, coords_value_mapping)

    naming_attrs = rename_columns(
        data, coords_cols, coords_defaults, coords_terminologies
    )
    attrs.update(naming_attrs)

    if coords_value_mapping and "unit" in coords_value_mapping:
        harmonize_units(data, unit_terminology=coords_value_mapping["unit"])
    else:
        harmonize_units(data)

    data = sort_columns_and_rows(data)

    data.attrs = attrs

    return data


def check_mandatory_dimensions(
    coords_cols: Dict[str, str],
    coords_defaults: Dict[str, Any],
):
    """Check if all mandatory dimensions are specified."""
    for coord in INTERCHANGE_FORMAT_MANDATORY_COLUMNS:
        if coord not in coords_cols and coord not in coords_defaults:
            logger.error(
                f"Mandatory dimension {coord} not found in coords_cols={coords_cols} or"
                f" coords_defaults={coords_defaults}."
            )
            raise ValueError(f"Mandatory dimension {coord} not defined.")


def check_overlapping_specifications(
    coords_cols: Dict[str, str],
    coords_defaults: Dict[str, Any],
):
    both = set(coords_cols.keys()).intersection(set(coords_defaults.keys()))
    if both:
        logger.error(
            f"{both!r} is given in coords_cols and coords_defaults, but"
            f" it must only be given in one of them."
        )
        raise ValueError(f"{both!r} given in coords_cols and coords_defaults.")


def read_csv(
    filepath_or_buffer,
    coords_cols: Dict[str, str],
) -> pd.DataFrame:

    na_values = [
        "nan",
        "NE",
        "-",
        "NA, NE",
        "NO,NE",
        "NA,NE",
        "NE,NO",
        "NE0",
        "NO, NE",
    ]
    read_data = pd.read_csv(filepath_or_buffer, na_values=na_values)

    # get all the columns that are actual data not metadata (usually the years)
    year_regex = re.compile(r"\d")
    year_cols = list(filter(year_regex.match, read_data.columns.values))

    # remove all non-numeric values from year columns
    # (what is left after mapping to nan when reading data)
    for col in year_cols:
        read_data[col] = read_data[col][
            pd.to_numeric(read_data[col], errors="coerce").notnull()
        ]

    # remove all cols not in the specification
    columns = read_data.columns.values
    read_data.drop(
        columns=list(set(columns) - set(coords_cols.values()) - set(year_cols)),
        inplace=True,
    )

    # check that all cols in the specification could be read
    missing = set(coords_cols.values()) - set(read_data.columns.values)
    if missing:
        logger.error(
            f"Column(s) {missing} specified in coords_cols, but not found in "
            f"the CSV file {filepath_or_buffer!r}."
        )
        raise ValueError(f"Columns {missing} not found in CSV.")

    return read_data


def spec_to_query_string(filter_spec: Dict[str, Union[list, Any]]) -> str:
    """Convert filter specification to query string.

    All column conditions in the filter are combined with &."""
    queries = []
    for col in filter_spec:
        if isinstance(filter_spec[col], list):
            itemlist = ", ".join((repr(x) for x in filter_spec[col]))
            filter_query = f"{col} in [{itemlist}]"
        else:
            filter_query = f"{col} == {filter_spec[col]!r}"
        queries.append(filter_query)

    return " & ".join(queries)


def filter_data(
    read_data: pd.DataFrame,
    filter_keep: Optional[Dict[str, Dict[str, Any]]] = None,
    filter_remove: Optional[Dict[str, Dict[str, Any]]] = None,
):
    # Filters for keeping data are combined with "or" so that
    # everything matching at least one rule is kept.
    if filter_keep:
        queries = []
        for filter_spec in filter_keep.values():
            q = spec_to_query_string(filter_spec)
            queries.append(f"({q})")
        query = " | ".join(queries)
        read_data.query(query, inplace=True)

    # Filters for removing data are negated and combined with "and" so that
    # only rows which don't match any rule are kept.
    if filter_remove:
        queries = []
        for filter_spec in filter_remove.values():
            q = spec_to_query_string(filter_spec)
            queries.append(f"~({q})")
        query = " & ".join(queries)
        read_data.query(query, inplace=True)

    read_data.reset_index(drop=True, inplace=True)


def add_dimensions_from_defaults(
    read_data: pd.DataFrame,
    coords_defaults: Dict[str, Any],
):
    if_columns = (
        INTERCHANGE_FORMAT_OPTIONAL_COLUMNS + INTERCHANGE_FORMAT_MANDATORY_COLUMNS
    )
    for coord in coords_defaults.keys():
        if coord in if_columns or coord.startswith(SEC_CATS_PREFIX):
            # add column to dataframe with default value
            read_data[coord] = coords_defaults[coord]
        else:
            raise ValueError(
                f"{coord!r} given in coords_defaults is unknown - prefix with "
                f"{SEC_CATS_PREFIX!r} to add a secondary category."
            )


def map_metadata(
    read_data: pd.DataFrame,
    coords_cols: Dict[str, str],
    meta_mapping: Dict[str, Union[str, Callable, dict]],
):
    """Map the metadata according to specifications given in meta_mapping."""

    # TODO: add additional mapping functions here
    mapping_functions = {
        "PRIMAP1": {
            "category": convert_ipcc_code_primap_to_primap2,
            "entity": convert_entity_gwp_primap,
        }
    }

    meta_mapping_df = {}
    # preprocess meta_mapping
    for column, mapping in meta_mapping.items():
        if column == "unit":  # handled in harmonize_units
            continue

        if isinstance(mapping, str) or callable(mapping):
            if isinstance(mapping, str):  # need to translate to function first
                try:
                    func = mapping_functions[mapping][column]
                except KeyError:
                    logger.error(
                        f"Unknown metadata mapping {mapping} for column {column}, "
                        f"known mappings are: {mapping_functions}."
                    )
                    raise ValueError(
                        f"Unknown metadata mapping {mapping} for column {column}."
                    )
            else:
                func = mapping

            values_to_map = read_data[coords_cols[column]].unique()
            values_mapped = map(func, values_to_map)
            meta_mapping_df[coords_cols[column]] = dict(
                zip(values_to_map, values_mapped)
            )

        else:
            meta_mapping_df[coords_cols[column]] = mapping

    read_data.replace(meta_mapping_df, inplace=True)


def rename_columns(
    read_data: pd.DataFrame,
    coords_cols: Dict[str, str],
    coords_defaults: Dict[str, Any],
    coords_terminologies: Dict[str, str],
) -> dict:
    """Rename columns to match PRIMAP2 specifications and generate the corresponding
    dataset-wide attrs for PRIMAP2."""

    attr_names = {"category": "cat", "scenario": "scen", "area": "area"}

    attrs = {}
    sec_cats = []
    coord_renaming = {}
    for coord in itertools.chain(coords_cols, coords_defaults):
        if coord in coords_terminologies:
            name = f"{coord} ({coords_terminologies[coord]})"
        else:
            name = coord

        if coord.startswith(SEC_CATS_PREFIX):
            name = name[len(SEC_CATS_PREFIX) :]
            sec_cats.append(name)
        elif coord in attr_names:
            attrs[attr_names[coord]] = name

        coord_renaming[coords_cols.get(coord, coord)] = name

    read_data.rename(columns=coord_renaming, inplace=True)

    if sec_cats:
        attrs["sec_cats"] = sec_cats

    return attrs


def harmonize_units(
    data: pd.DataFrame,
    unit_terminology: str = "PRIMAP1",
    unit_col: str = "unit",
    entity_col: str = "entity",
    data_col_regex_str: str = r"\d",
):
    """
    Harmonize the units of the input data. For each entity, convert
    all time series to the same unit (the unit that occurs first). The unit strings
    are converted to PRIMAP2 style. The input data unit style is defined via the
    "unit_terminology" parameter.

    Parameters
    ----------
    data: pd.DataFrame
        data for which the units should be harmonized
    unit_terminology: str
        unit terminology of input data. This defines how the units are written. E.g.
        with entity given or without, or with or without timeframe.
        Default (and currently only option): PRIMAP1
    unit_col: str
        column name for unit column. Default: "unit"
    entity_col: str
        column name for entity column. Default: "entity"
    data_col_regex_str: str
        regular expression to match the columns with data. default: matches digits only

    Returns
    -------
    obj: pd.DataFrame
        pandas DataFrame with the converted data

    """
    # we need to convert the units to primap2 (pint) style and
    # then convert the data such that we have one unit per entity
    # first convert the units such that pint understands them
    if unit_terminology == "PRIMAP1":
        convert_dataframe_units_primap_to_primap2(
            data, unit_col="unit", entity_col="entity"
        )
    else:
        logger.error("Unknown unit terminology " + unit_terminology + ".")
        raise ValueError("Unknown unit terminology " + unit_terminology + ".")

    # find the data columns
    data_col_regex = re.compile(data_col_regex_str)
    data_cols = list(filter(data_col_regex.match, data.columns.values))

    # get entities from read data
    entities = data[entity_col].unique()
    for entity in entities:
        # get all units for this entity
        data_this_entity = data.loc[data[entity_col] == entity]
        units_this_entity = data_this_entity[unit_col].unique()
        # print(units_this_entity)
        if len(units_this_entity) > 1:
            # need unit conversion. convert to first unit (base units have second as
            # time that is impractical)
            unit_to = units_this_entity[0]
            # print("unit_to: " + unit_to)
            for unit in units_this_entity[1:]:
                # print(unit)
                unit_pint = ureg[unit]
                unit_pint = unit_pint.to(unit_to)
                # print(unit_pint)
                factor = unit_pint.magnitude
                # print(factor)
                mask = (data[entity_col] == entity) & (data[unit_col] == unit)
                data.loc[mask, data_cols] *= factor
                data.loc[mask, unit_col] = unit_to


def sort_columns_and_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Sort the data.

    The columns are ordered according to the order in
    INTERCHANGE_FORMAT_COLUMN_ORDER, with secondary categories alphabetically after
    the category and all date columns in order at the end.

    The rows are ordered by values of the non-date columns.
    """
    columns = data.columns.values

    date_regex = re.compile(r"\d")
    date_cols = list(filter(date_regex.match, columns))

    other_cols = set(columns) - set(date_cols)

    cols_sorted = []
    for col in INTERCHANGE_FORMAT_COLUMN_ORDER:
        for ocol in other_cols:
            if ocol == col or ocol.startswith(f"{col} ("):
                cols_sorted.append(ocol)
                other_cols.remove(ocol)
                break

    cols_sorted += list(sorted(other_cols))

    data = data[cols_sorted + list(sorted(date_cols))]

    data.sort_values(by=cols_sorted, inplace=True)
    data.reset_index(inplace=True, drop=True)

    return data


def convert_entity_gwp_primap(entity_pm1: str) -> str:
    """
    This function transforms PRIMAP1 style entity names into PRIMAP2 style variable
    names. The transformation only considers the GWP, currently the variable itself is
    unchanged.

    Currently the function uses a limited set of GWP values (defined in gwp_mapping) and
    works on a limited set of variables (defined in entities_gwp).

    Parameters
    entity: str
        entity to process

    Returns
    -------
    entity: str
        entity in PRIMAP2 format

    """

    entities_gwp = [
        "KYOTOGHG",
        "HFCS",
        "PFCS",
        "FGASES",
        "OTHERHFCS CO2EQ",
        "OTHERHFCS",
        "OTHERPFCS",
    ]

    # define the mapping of PRIMAP GWP specifications to PRIMAP2 GWP specification
    # no GWP given will be mapped to SAR
    gwp_mapping = {
        "SAR": "SARGWP100",
        "AR4": "AR4GWP100",
        "AR5": "AR5GWP100",
        "AR5CCF": "AR5CCFGWP100",  # not sure if implemented in scmdata units
    }

    # build regexp to match the GWP conversion variables
    regexp_str = "("
    for current_entity in entities_gwp:
        regexp_str = regexp_str + current_entity + "|"
    regexp_str = regexp_str[0:-1] + ")"

    # build regexp to match the GWPs
    regexp_str_gwps = "("
    for mapping in gwp_mapping.keys():
        regexp_str_gwps = regexp_str_gwps + mapping + "|"
    regexp_str_gwps = regexp_str_gwps[0:-1] + ")$"

    # check if entity in entities_gwp
    found = re.match(regexp_str, entity_pm1)
    if found is None:
        # not a basket entity which uses GWPs
        entity_pm2 = entity_pm1
    else:
        # check if GWP information present in entity
        match = re.search(regexp_str_gwps, entity_pm1)
        if match is None:
            # SAR GWPs are default in PRIMAP
            entity_pm2 = entity_pm1 + " (" + gwp_mapping["SAR"] + ")"
        else:
            gwp_out = match.group(0)
            # in this case the entity has to be replaced as well
            match = re.search(".*(?=" + gwp_out + ")", entity_pm1)
            if match.group() == "":
                logger.error(
                    "Confused: could not find entity which should be there."
                    " This indicates a bug in this function."
                )
                raise RuntimeError(
                    "Confused: could not find entity which should be there."
                    " This indicates a bug in this function."
                )
            else:
                entity_pm2 = match.group(0) + " (" + gwp_mapping[gwp_out] + ")"

    return entity_pm2


def metadata_for_variable(unit: str, variable: str) -> dict:
    """
    This function takes GWP information from the variable name in primap2 style (if
    present). Information is returned as attrs dict.

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

    attrs = {"units": unit}

    regex_gwp = r"\s\(([A-Za-z0-9]*)\)$"
    regex_variable = r"^(.*)\s\([a-zA-z0-9]*\)$"

    gwp = re.findall(regex_gwp, variable)
    if gwp:
        attrs["gwp_context"] = gwp[0]
        entity = re.findall(regex_variable, variable)
        if not entity:
            logger.error("Can't extract entity from " + variable)
            raise ValueError("Can't extract entity from " + variable)
        attrs["entity"] = entity[0]
    else:
        attrs["entity"] = variable
    return attrs


def from_interchange_format(
    data: pd.DataFrame,
    attrs: Optional[dict] = None,
    time_col_regex: str = r"\d",
    max_array_size: int = 1024 * 1024 * 1024,
) -> xr.Dataset:
    """
    This function converts an interchange format DataFrame to an PRIMAP2 xarray data
    structure. All column names and attrs are expected to be already in PRIMAP2 format
    as defined for the interchange format. The attrs dict is given explicitly as the
    attrs functionality in pandas is experimental.

    Parameters
    ----------
    data: pd.DataFrame
        pandas DataFrame in PRIMAP2 interchange format.
    attrs: dict, optional
        attrs dict as defined for PRIMAP2 xarray datasets. Default: use data.attrs.
    time_col_regex: str, optional
        Regular expression matching the columns which will form the time
    max_array_size: int, optional
        Maximum permitted projected array size. Larger sizes will raise an exception.
        Default: 1 G, corresponding to about 4 GB of memory usage.

    Returns
    -------
    obj: xr.Dataset
        xr dataset with the converted data

    """
    if attrs is None:
        attrs = data.attrs

    columns = data.columns.values
    # find the time columns
    time_cols = list(filter(re.compile(time_col_regex).match, columns))

    # convert to xarray
    data_xr = data.to_xarray()
    index_cols = list(set(columns) - set(time_cols) - {"unit"})
    data_xr = data_xr.set_index({"index": index_cols})
    # take the units out as they increase dimensionality and we have only one unit per
    # entity/variable
    units = data_xr["unit"].dropna("index")
    del data_xr["unit"]

    # check resulting shape to estimate memory consumption
    shape = []
    for col in index_cols:
        shape.append(len(np.unique(data_xr[col])))
    array_size = np.product(shape)
    logger.debug(f"Expected array shape: {shape}, resulting in size {array_size:,}.")
    if array_size > max_array_size:
        logger.error(
            f"Array with {len(shape)-1} dimensions will have a size of {array_size:,} "
            f"due to the shape {shape}. Aborting to avoid out-of-memory errors. To "
            f"continue, raise max_array_size (currently {max_array_size:,})."
        )
        raise ValueError(
            f"Resulting array too large: {array_size:,} > {max_array_size:,}. To "
            f"continue, raise max_array_size."
        )

    # convert time to dimension and entity to variables.
    data_xr = dates_to_dimension(data_xr).to_dataset("entity")

    # fill the entity/variable attributes
    for variable in data_xr:
        csv_units = np.unique(units.loc[{"entity": variable}])
        if len(csv_units) > 1:
            logger.error(
                f"More than one unit for {variable!r}: {csv_units!r}. "
                + "There is an error in the unit harmonization."
            )
            raise ValueError(f"More than one unit for {variable!r}: {csv_units!r}.")
        data_xr[variable].attrs = metadata_for_variable(csv_units[0], variable)

    # add the dataset wide attributes
    data_xr.attrs = attrs

    data_xr = data_xr.pr.quantify()

    data_xr.pr.ensure_valid()
    return data_xr
