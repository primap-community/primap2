import datetime
import itertools
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
import pint
from loguru import logger

from .. import _selection
from .._units import ureg
from . import _conversion
from ._interchange_format import (
    INTERCHANGE_FORMAT_COLUMN_ORDER,
    INTERCHANGE_FORMAT_MANDATORY_COLUMNS,
    INTERCHANGE_FORMAT_OPTIONAL_COLUMNS,
)

SEC_CATS_PREFIX = "sec_cats__"

BASKET_UNITS = [
    "KYOTOGHG",
    "FGASES",
    "HFCS",
    "PFCS",
    "OTHERHFCS",
    "OTHERPFCS",
]


def convert_long_dataframe_if(
    data_long: pd.DataFrame,
    *,
    coords_cols: dict[str, str],
    add_coords_cols: None | dict[str, list[str]] = None,
    coords_defaults: None | dict[str, Any] = None,
    coords_terminologies: dict[str, str],
    coords_value_mapping: None | dict[str, Any] = None,
    coords_value_filling: None | dict[str, dict[str, dict]] = None,
    filter_keep: None | dict[str, dict[str, Any]] = None,
    filter_remove: None | dict[str, dict[str, Any]] = None,
    meta_data: None | dict[str, Any] = None,
    time_format: str = "%Y-%m-%d",
    convert_str: bool | dict[str, float] = True,
    copy_df: bool = True,
) -> pd.DataFrame:
    """convert a DataFrame in long (tidy) format into the PRIMAP2 interchange format.

    Columns can be renamed or filled with default values to match the PRIMAP2
    structure. Where we refer to "dimensions" in the parameter description below we
    mean the basic dimension names without the added terminology (e.g. "area" not
    "area (ISO3)"). The terminology information will be added by this function. You can
    not use the short dimension names in the attributes (e.g. "cat" instead of
    "category").

    Parameters
    ----------
    data_long: str, pd.DataFrame
        Long format DataFrame file which will be converted.

    coords_cols : dict
        Dict where the keys are column names in the files to be read and the value is
        the dimension in PRIMAP2. To specify the data column containing the observable,
        use the "data" key. For secondary categories use a ``sec_cats__`` prefix.

    add_coords_cols : dict, optional
        Dict where the keys are PRIMAP2 additional coordinate names and the values are
        lists with two elements where the first is the column in the dataframe to be
        converted and the second is the primap2 dimension for the coordinate (e.g.
        ``category`` for a ``category_name`` coordinate).

    coords_defaults : dict, optional
        Dict for default values of coordinates / dimensions not given in the csv files.
        The keys are the dimension names and the values are the values for
        the dimensions. For secondary categories use a ``sec_cats__`` prefix.

    coords_terminologies : dict
        Dict defining the terminologies used for the different coordinates (e.g. ISO3
        for area). Only possible coordinates here are: area, category, scenario,
        entity, and secondary categories. For secondary categories use a ``sec_cats__``
        prefix. All entries different from "area", "category", "scenario", "entity",
        and ``sec_cats__<name>`` will raise a ValueError.

    coords_value_mapping : dict, optional
        A dict with primap2 dimension names as keys. Values are dicts with input values
        as keys and output values as values. A standard use case is to map gas names
        from input data to the standardized names used in primap2.
        Alternatively a value can also be a function which transforms one CSV metadata
        value into the new metadata value.
        A third possibility is to give a string as a value, which defines a rule for
        translating metadata values. For the "category", "entity", and "unit" columns,
        the rule "PRIMAP1" is available, which translates from PRIMAP1 metadata to
        PRIMAP2 metadata.

    coords_value_filling : dict, optional
        A dict with primap2 dimension names as keys. These are the target columns where
        values will be filled (or replaced). Vales are dicts with primap2 dimension
        names as keys. These are the source columns. The values are dicts with source
        value - target value mappings.
        The value filling can do everything that the value mapping can, but while
        mapping can only replace values within a column using information from that
        column, the filing function can also fill or replace data based on values from
        a different column.
        This can be used to e.g. fill missing category codes based on category names or
        to replace category codes which do not meet the terminology using the category
        names.

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
        `data format documentation <https://primap2.readthedocs.io/en/stable/\
data_format_details.html#dataset-attributes>`_.

    time_format : str, optional (default: "%Y-%m-%d")
        strftime style format used to format the time information for the data columns
        in the interchange format.
        Default: "%F", i.e. the ISO 8601 date format.

    convert_str : bool or dict, optional (default: True)
        If set to false, string values in the data columns will be kept.
        If set to true they will be converted to np.nan or 0 following default rules.
        If a dict is given mapping will be as given in the dict for values present in
        the dict and default as in parse_code for all other values

    copy_df : bool, optional (default: True)
        If set to true, a copy of the input DataFrame is made to keep the input as is.
        This negatively impacts speed. If set to false the input DataFrame will be
        altered but performance will be better

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
    if add_coords_cols is None:
        add_coords_cols = {}
    if meta_data is None:
        attrs = {}
    else:
        attrs = meta_data.copy()

    check_mandatory_dimensions(coords_cols, coords_defaults)
    check_overlapping_specifications(coords_cols, coords_defaults)
    if add_coords_cols:
        check_overlapping_specifications_add_cols(coords_cols, add_coords_cols)

    # if desired make a copy to keep input dataframe unchanged
    if copy_df:
        data_copy = data_long.copy(deep=True)
    else:
        data_copy = data_long

    filter_data(data_copy, filter_keep, filter_remove)

    add_dimensions_from_defaults(
        data_copy, coords_defaults, additional_allowed_coords=["time"]
    )

    naming_attrs = rename_columns(
        data_copy, coords_cols, add_coords_cols, coords_defaults, coords_terminologies
    )
    attrs.update(naming_attrs)

    if convert_str:
        # get data columns (just one as we have long format)
        data_cols = ["data"]
        # find all string values
        str_values = find_str_values_in_data(data_copy, data_cols)
        # create replacement dict
        str_repl_dict = create_str_replacement_dict(str_values, convert_str)
        replace_values(data_copy, data_cols, str_repl_dict)

    additional_coordinates = additional_coordinate_metadata(
        add_coords_cols, coords_cols, coords_terminologies
    )

    if coords_value_mapping is not None:
        map_metadata(data_copy, attrs=attrs, meta_mapping=coords_value_mapping)

    if coords_value_filling is not None:
        data_copy = fill_from_other_col(
            data_copy, attrs=attrs, coords_value_filling=coords_value_filling
        )

    coords = list(set(data_copy.columns.values) - {"data"})

    harmonize_units(data_copy, dimensions=coords, attrs=attrs)

    data_copy["time"] = pd.to_datetime(data_copy["time"], format=time_format)

    data, coords = long_to_wide(data_copy, time_format=time_format)

    data, coords = sort_columns_and_rows(data, dimensions=coords)
    dims = coords.copy()
    for add_coord in add_coords_cols.keys():
        dims.remove(add_coord)

    data.attrs = interchange_format_attrs_dict(
        xr_attrs=attrs,
        time_format=time_format,
        dimensions=dims,
        additional_coordinates=additional_coordinates,
    )

    return data


def read_long_csv_file_if(
    filepath_or_buffer: str | Path | IO,
    *,
    coords_cols: dict[str, str],
    add_coords_cols: None | dict[str, list[str]] = None,
    coords_defaults: None | dict[str, Any] = None,
    coords_terminologies: dict[str, str],
    coords_value_mapping: None | dict[str, Any] = None,
    coords_value_filling: None | dict[str, dict[str, dict]] = None,
    filter_keep: None | dict[str, dict[str, Any]] = None,
    filter_remove: None | dict[str, dict[str, Any]] = None,
    meta_data: None | dict[str, Any] = None,
    time_format: str = "%Y-%m-%d",
    convert_str: bool | dict[str, float] = True,
) -> pd.DataFrame:
    """Read a CSV file in long (tidy) format into the PRIMAP2 interchange format.

    Columns can be renamed or filled with default values to match the PRIMAP2
    structure. Where we refer to "dimensions" in the parameter description below we
    mean the basic dimension names without the added terminology (e.g. "area" not
    "area (ISO3)"). The terminology information will be added by this function. You can
    not use the short dimension names in the attributes (e.g. "cat" instead of
    "category").

    Parameters
    ----------
    filepath_or_buffer: str, pathlib.Path, or file-like
        Long CSV file which will be read.

    coords_cols : dict
        Dict where the keys are column names in the files to be read and the value is
        the dimension in PRIMAP2. To specify the data column containing the observable,
        use the "data" key. For secondary categories use a ``sec_cats__`` prefix.

    add_coords_cols : dict, optional
        Dict where the keys are PRIMAP2 additional coordinate names and the values are
        lists with two elements where the first is the column in the csv file to be
        read and the second is the primap2 dimension for the coordinate (e.g.
        ``category`` for a ``category_name`` coordinate).

    coords_defaults : dict, optional
        Dict for default values of coordinates / dimensions not given in the csv files.
        The keys are the dimension names and the values are the values for
        the dimensions. For secondary categories use a ``sec_cats__`` prefix.

    coords_terminologies : dict
        Dict defining the terminologies used for the different coordinates (e.g. ISO3
        for area). Only possible coordinates here are: area, category, scenario,
        entity, and secondary categories. For secondary categories use a ``sec_cats__``
        prefix. All entries different from "area", "category", "scenario", "entity",
        and ``sec_cats__<name>`` will raise a ValueError.

    coords_value_mapping : dict, optional
        A dict with primap2 dimension names as keys. Values are dicts with input values
        as keys and output values as values. A standard use case is to map gas names
        from input data to the standardized names used in primap2.
        Alternatively a value can also be a function which transforms one CSV metadata
        value into the new metadata value.
        A third possibility is to give a string as a value, which defines a rule for
        translating metadata values. For the "category", "entity", and "unit" columns,
        the rule "PRIMAP1" is available, which translates from PRIMAP1 metadata to
        PRIMAP2 metadata.

    coords_value_filling : dict, optional
        A dict with primap2 dimension names as keys. These are the target columns where
        values will be filled (or replaced). Vales are dicts with primap2 dimension
        names as keys. These are the source columns. The values are dicts with source
        value - target value mappings.
        The value filling can do everything that the value mapping can, but while
        mapping can only replace values within a column using information from that
        column, the filing function can also fill or replace data based on values from
        a different column.
        This can be used to e.g. fill missing category codes based on category names or
        to replace category codes which do not meet the terminology using the category
        names.

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
        `data format documentation <https://primap2.readthedocs.io/en/stable/\
data_format_details.html#dataset-attributes>`_.

    time_format : str, optional
        strftime style format used to format the time information for the data columns
        in the interchange format.
        Default: "%F", i.e. the ISO 8601 date format.

    convert_str : bool or dict, optional (default: True)
        If set to false, string values in the data columns will be kept.
        If set to true they will be converted to np.nan or 0 following default rules.
        If a dict is given mapping will be as given in the dict for values present in
        the dict and default as in parse_code for all other values

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
    check_mandatory_dimensions(coords_cols, coords_defaults)
    check_overlapping_specifications(coords_cols, coords_defaults)
    if add_coords_cols:
        check_overlapping_specifications_add_cols(coords_cols, add_coords_cols)

    data_long = read_long_csv(filepath_or_buffer, coords_cols, add_coords_cols)

    return convert_long_dataframe_if(
        data_long=data_long,
        coords_cols=coords_cols,
        add_coords_cols=add_coords_cols,
        coords_defaults=coords_defaults,
        coords_terminologies=coords_terminologies,
        coords_value_mapping=coords_value_mapping,
        coords_value_filling=coords_value_filling,
        filter_keep=filter_keep,
        filter_remove=filter_remove,
        meta_data=meta_data,
        time_format=time_format,
        convert_str=convert_str,
        copy_df=False,
    )


def long_to_wide(
    data_long: pd.DataFrame, *, time_format: str
) -> tuple[pd.DataFrame, list[str]]:
    data_long["time"] = data_long["time"].dt.strftime(time_format)

    coords = list(set(data_long.columns.values) - {"data", "time"})

    # unit is neither a coordinate nor a data column, so has to be handled separately
    unit = data_long[coords].drop_duplicates()
    coords.remove("unit")
    unit.index = pd.MultiIndex.from_frame(unit[coords])

    series = data_long["data"]
    series.index = pd.MultiIndex.from_frame(data_long[[*coords, "time"]])
    data = series.unstack("time")

    data["unit"] = unit["unit"]

    data.reset_index(inplace=True)
    data.columns.name = None

    return data, [*coords, "unit"]


def convert_wide_dataframe_if(
    data_wide: pd.DataFrame,
    *,
    coords_cols: dict[str, str],
    add_coords_cols: None | dict[str, list[str]] = None,
    coords_defaults: None | dict[str, Any] = None,
    coords_terminologies: dict[str, str],
    coords_value_mapping: None | dict[str, Any] = None,
    coords_value_filling: None | dict[str, dict[str, dict]] = None,
    filter_keep: None | dict[str, dict[str, Any]] = None,
    filter_remove: None | dict[str, dict[str, Any]] = None,
    meta_data: None | dict[str, Any] = None,
    time_format: str = "%Y",
    time_cols: None | list = None,
    convert_str: bool | dict[str, float] = True,
    copy_df: bool = False,
) -> pd.DataFrame:
    """
    Convert a DataFrame in wide format into the PRIMAP2 interchange format.

    Columns can be renamed or filled with default values to match the PRIMAP2
    structure. Where we refer to "dimensions" in the parameter description below we
    mean the basic dimension names without the added terminology (e.g. "area" not
    "area (ISO3)"). The terminology information will be added by this function. You can
    not use the short dimension names in the attributes (e.g. "cat" instead of
    "category").

    TODO: Currently duplicate data points will not be detected.

    TODO: enable filtering through query strings

    TODO: enable specification of the entity terminology

    Parameters
    ----------
    data_wide: pd.DataFrame
        Wide DataFrame which will be converted.

    coords_cols : dict
        Dict where the keys are PRIMAP2 dimension names and the values are column
        names in the dataframe to be converted.
        For secondary categories use a ``sec_cats__`` prefix.

    add_coords_cols : dict, optional
        Dict where the keys are PRIMAP2 additional coordinate names and the values are
        lists with two elements where the first is the column in the dataframe to be
        converted and the second is the primap2 dimension for the coordinate (e.g.
        ``category`` for a ``category_name`` coordinate.

    coords_defaults : dict, optional
        Dict for default values of coordinates / dimensions not given in the dataframe.
        The keys are the dimension names and the values are the values for
        the dimensions. For secondary categories use a ``sec_cats__`` prefix.

    coords_terminologies : dict
        Dict defining the terminologies used for the different coordinates (e.g. ISO3
        for area). Only possible coordinates here are: area, category, scenario,
        entity, and secondary categories. For secondary categories use a ``sec_cats__``
        prefix. All entries different from "area", "category", "scenario", "entity",
        and ``sec_cats__<name>`` will raise a ValueError.

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

    coords_value_filling : dict, optional
        A dict with primap2 dimension names as keys. These are the target columns where
        values will be filled (or replaced). Vales are dicts with primap2 dimension
        names as keys. These are the source columns. The values are dicts with source
        value - target value mappings.
        The value filling can do everything that the value mapping can, but while
        mapping can only replace values within a column using information from that
        column, the filing function can also fill or replace data based on values from
        a different column.
        This can be used to e.g. fill missing category codes based on category names or
        to replace category codes which do not meet the terminology using the category
        names.

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
        `data format documentation <https://primap2.readthedocs.io/en/stable/\
data_format_details.html#dataset-attributes>`_.

    time_format : str
        str with strftime style format used to parse the time information for
        the data columns.
        Default: "%Y", which will match years.

    time_cols : list, optional
        List of column names which contain the data for each time point. If not given
        cols will be inferred using time_format.

    convert_str : bool or dict, optional (default: True)
        If set to false, string values in the data columns will be kept.
        If set to true they will be converted to np.nan or 0 following default rules.
        If a dict is given mapping will be as given in the dict for values present in
        the dict and default as in parse_code for all other values

    copy_df : bool, optional (default: True)
        If set to true, a copy of the input DataFrame is made to keep the input as is.
        This negatively impacts speed. If set to false the input DataFrame will be
        altered but performance will be better

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
    if add_coords_cols is None:
        add_coords_cols = {}
    if meta_data is None:
        attrs = {}
    else:
        attrs = meta_data.copy()

    check_mandatory_dimensions(coords_cols, coords_defaults)
    check_overlapping_specifications(coords_cols, coords_defaults)
    if add_coords_cols:
        check_overlapping_specifications_add_cols(coords_cols, add_coords_cols)

    # get all the columns that are actual data not metadata (usually the years)
    if time_cols is None:
        time_columns = [
            col
            for col in data_wide.columns.values
            if matches_time_format(col, time_format)
        ]
    else:
        time_columns = time_cols

    # if desired make a copy of the data to not alter the input data
    if copy_df:
        data_if = data_wide.copy(deep=True)
    else:
        data_if = data_wide

    filter_data(data_if, filter_keep, filter_remove)

    if convert_str:
        # get data columns (just one as we have long format)
        # find all string values
        str_values = find_str_values_in_data(data_if, time_columns)
        # create replacement dict
        str_repl_dict = create_str_replacement_dict(str_values, convert_str)
        replace_values(data_if, time_columns, str_repl_dict)

    add_dimensions_from_defaults(data_if, coords_defaults)

    naming_attrs = rename_columns(
        data_if, coords_cols, add_coords_cols, coords_defaults, coords_terminologies
    )
    attrs.update(naming_attrs)

    additional_coordinates = additional_coordinate_metadata(
        add_coords_cols, coords_cols, coords_terminologies
    )

    if coords_value_mapping is not None:
        map_metadata(data_if, attrs=attrs, meta_mapping=coords_value_mapping)

    if coords_value_filling is not None:
        data_if = fill_from_other_col(
            data_if, attrs=attrs, coords_value_filling=coords_value_filling
        )

    coords = list(set(data_if.columns.values) - set(time_columns))

    harmonize_units(data_if, dimensions=coords, attrs=attrs)

    data_if, coords = sort_columns_and_rows(data_if, dimensions=coords)
    dims = coords.copy()
    for add_coord in add_coords_cols.keys():
        dims.remove(add_coord)

    data_if.attrs = interchange_format_attrs_dict(
        xr_attrs=attrs,
        time_format=time_format,
        dimensions=dims,
        additional_coordinates=additional_coordinates,
    )

    return data_if


def read_wide_csv_file_if(
    filepath_or_buffer: str | Path | IO,
    *,
    coords_cols: dict[str, str],
    add_coords_cols: None | dict[str, list[str]] = None,
    coords_defaults: None | dict[str, Any] = None,
    coords_terminologies: dict[str, str],
    coords_value_mapping: None | dict[str, Any] = None,
    coords_value_filling: None | dict[str, dict[str, dict]] = None,
    filter_keep: None | dict[str, dict[str, Any]] = None,
    filter_remove: None | dict[str, dict[str, Any]] = None,
    meta_data: None | dict[str, Any] = None,
    time_format: str = "%Y",
    convert_str: bool | dict[str, float] = True,
) -> pd.DataFrame:
    """Read a CSV file in wide format into the PRIMAP2 interchange format.

    Columns can be renamed or filled with default values to match the PRIMAP2
    structure. Where we refer to "dimensions" in the parameter description below we
    mean the basic dimension names without the added terminology (e.g. "area" not
    "area (ISO3)"). The terminology information will be added by this function. You can
    not use the short dimension names in the attributes (e.g. "cat" instead of
    "category").

    TODO: Currently duplicate data points will not be detected.

    TODO: enable filtering through query strings

    TODO: enable specification of the entity terminology

    Parameters
    ----------
    filepath_or_buffer: str, pathlib.Path, or file-like
        Wide CSV file which will be read.

    coords_cols : dict
        Dict where the keys are PRIMAP2 dimensions and the values are column names in
        the files to be read. For secondary categories use a ``sec_cats__`` prefix.

    add_coords_cols : dict, optional
        Dict where the keys are PRIMAP2 additional coordinate names and the values are
        lists with two elements where the first is the column in the csv file to be
        read and the second is the primap2 dimension for the coordinate (e.g.
        ``category`` for a ``category_name`` coordinate.

    coords_defaults : dict, optional
        Dict for default values of coordinates / dimensions not given in the csv files.
        The keys are the dimension names and the values are the values for
        the dimensions. For secondary categories use a ``sec_cats__`` prefix.

    coords_terminologies : dict
        Dict defining the terminologies used for the different coordinates (e.g. ISO3
        for area). Only possible coordinates here are: area, category, scenario,
        entity, and secondary categories. For secondary categories use a ``sec_cats__``
        prefix. All entries different from "area", "category", "scenario", "entity",
        and ``sec_cats__<name>`` will raise a ValueError.

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

    coords_value_filling : dict, optional
        A dict with primap2 dimension names as keys. These are the target columns where
        values will be filled (or replaced). Vales are dicts with primap2 dimension
        names as keys. These are the source columns. The values are dicts with source
        value - target value mappings.
        The value filling can do everything that the value mapping can, but while
        mapping can only replace values within a column using information from that
        column, the filing function can also fill or replace data based on values from
        a different column.
        This can be used to e.g. fill missing category codes based on category names or
        to replace category codes which do not meet the terminology using the category
        names.

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
        `data format documentation <https://primap2.readthedocs.io/en/stable/\
data_format_details.html#dataset-attributes>`_.

    time_format : str, optional
        strftime style format used to parse the time information for the data columns.
        Default: "%Y", which will match years.

    convert_str : bool or dict, optional (default: True)
        If set to false, string values in the data columns will be kept.
        If set to true they will be converted to np.nan or 0 following default rules.
        If a dict is given mapping will be as given in the dict for values present in
        the dict and default as in parse_code for all other values

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

    check_mandatory_dimensions(coords_cols, coords_defaults)
    check_overlapping_specifications(coords_cols, coords_defaults)
    if add_coords_cols:
        check_overlapping_specifications_add_cols(coords_cols, add_coords_cols)

    data, time_columns = read_wide_csv(
        filepath_or_buffer,
        coords_cols,
        add_coords_cols=add_coords_cols,
        time_format=time_format,
    )

    data = convert_wide_dataframe_if(
        data,
        coords_cols=coords_cols,
        add_coords_cols=add_coords_cols,
        coords_defaults=coords_defaults,
        coords_terminologies=coords_terminologies,
        coords_value_mapping=coords_value_mapping,
        coords_value_filling=coords_value_filling,
        filter_keep=filter_keep,
        filter_remove=filter_remove,
        meta_data=meta_data,
        time_format=time_format,
        time_cols=time_columns,
        convert_str=convert_str,
        copy_df=False,
    )

    return data


def interchange_format_attrs_dict(
    *,
    xr_attrs: dict,
    time_format: str,
    dimensions,
    additional_coordinates: None | dict = None,
) -> dict:
    metadata = {
        "attrs": xr_attrs,
        "time_format": time_format,
        "dimensions": {"*": dimensions.copy()},
    }

    if additional_coordinates:
        metadata["additional_coordinates"] = additional_coordinates

    return metadata


def additional_coordinate_metadata(
    add_coords_cols: dict[str, list[str]],
    coords_cols: dict[str, str],
    coords_terminologies: dict[str, str],
) -> dict:
    """Create the `additional_coordinates` dict and do a few consistency checks"""

    additional_coordinates = {}
    for coord in add_coords_cols:
        if coord in coords_terminologies:
            logger.error(
                f"Additional coordinate {coord} has terminology definition. "
                f"This is currently not supported by PRIMAP2."
            )
            raise ValueError(
                f"Additional coordinate {coord} has terminology definition. "
                f"This is currently not supported by PRIMAP2."
            )

        if add_coords_cols[coord][1] not in coords_cols:
            logger.error(
                f"Additional coordinate {coord} refers to unknown coordinate "
                f"{add_coords_cols[coord][1]}. "
            )
            raise ValueError(
                f"Additional coordinate {coord} refers to unknown coordinate "
                f"{add_coords_cols[coord][1]}. "
            )

        if add_coords_cols[coord][1] in coords_terminologies:
            additional_coordinates[coord] = (
                f"{add_coords_cols[coord][1]} "
                f"({coords_terminologies[add_coords_cols[coord][1]]})"
            )
        else:
            additional_coordinates[coord] = add_coords_cols[coord][1]

    return additional_coordinates


def check_mandatory_dimensions(
    coords_cols: dict[str, str],
    coords_defaults: dict[str, Any],
):
    """Check if all mandatory dimensions are specified."""
    for coord in INTERCHANGE_FORMAT_MANDATORY_COLUMNS:
        if coord not in coords_cols and coord not in coords_defaults:
            logger.error(
                f"Mandatory dimension {coord!r} not found in coords_cols={coords_cols}"
                f" or coords_defaults={coords_defaults}."
            )
            raise ValueError(f"Mandatory dimension {coord!r} not defined.")


def check_overlapping_specifications(
    coords_cols: dict[str, str],
    coords_defaults: dict[str, Any],
):
    both = set(coords_cols.keys()).intersection(set(coords_defaults.keys()))
    if both:
        logger.error(
            f"{both!r} is given in coords_cols and coords_defaults, but"
            f" it must only be given in one of them."
        )
        raise ValueError(f"{both!r} given in coords_cols and coords_defaults.")


def check_overlapping_specifications_add_cols(
    coords_cols: dict[str, str],
    add_coords_cols: dict[str, Any],
):
    cols_add = [val[0] for val in add_coords_cols.values()]
    both = set(coords_cols.values()).intersection(set(cols_add))
    if both:
        logger.error(
            f"columns {both!r} used for dimensions and additional coordinates, but"
            f" should be used in only one of them."
        )
        raise ValueError(f"{both!r} given in coords_cols and add_coords_cols.")


def matches_time_format(value: str, time_format: str) -> bool:
    try:
        datetime.datetime.strptime(value, time_format)
        return True
    except ValueError:
        return False


def read_wide_csv(
    filepath_or_buffer,
    coords_cols: dict[str, str],
    add_coords_cols: None | dict[str, list[str]] = None,
    time_format: str = "%Y",
) -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(
        filepath_or_buffer,
    )

    # get all the columns that are actual data not metadata (usually the years)
    time_cols = [
        col for col in data.columns.values if matches_time_format(col, time_format)
    ]

    # remove all cols not in the specification
    columns = data.columns.values
    if add_coords_cols:
        add_coords_col_names = {value[0] for value in add_coords_cols.values()}
    else:
        add_coords_col_names = set()

    data.drop(
        columns=list(
            set(columns)
            - set(coords_cols.values())
            - add_coords_col_names
            - set(time_cols)
        ),
        inplace=True,
    )

    # check that all cols in the specification could be read
    missing = set(coords_cols.values()) - set(data.columns.values)
    if missing:
        logger.error(
            f"Column(s) {missing} specified in coords_cols, but not found in "
            f"the CSV file {filepath_or_buffer!r}."
        )
        raise ValueError(f"Columns {missing} not found in CSV.")

    return data, time_cols


def read_long_csv(
    filepath_or_buffer,
    coords_cols: dict[str, str],
    add_coords_cols: None | dict[str, list[str]] = None,
) -> pd.DataFrame:
    if "data" not in coords_cols.keys():
        raise ValueError(
            "No data column in the CSV specified in coords_cols, so nothing to read."
        )

    if "time" in coords_cols:
        parse_dates = [coords_cols["time"]]
    else:
        parse_dates = False

    if add_coords_cols:
        add_coords_col_names = {value[0] for value in add_coords_cols.values()}
    else:
        add_coords_col_names = set()

    usecols = list(coords_cols.values()) + list(add_coords_col_names)

    data = pd.read_csv(
        filepath_or_buffer,
        parse_dates=parse_dates,
        usecols=usecols,
    )

    return data


def spec_to_query_string(filter_spec: dict[str, Any]) -> str:
    """Convert filter specification to query string.

    All column conditions in the filter are combined with &."""
    queries = []
    for col in filter_spec:
        if isinstance(filter_spec[col], list):
            itemlist = ", ".join(repr(x) for x in filter_spec[col])
            filter_query = f"`{col}` in [{itemlist}]"
        else:
            filter_query = f"`{col}` == {filter_spec[col]!r}"
        queries.append(filter_query)

    return " & ".join(queries)


def filter_data(
    data: pd.DataFrame,
    filter_keep: None | dict[str, dict[str, Any]] = None,
    filter_remove: None | dict[str, dict[str, Any]] = None,
):
    # Filters for keeping data are combined with "or" so that
    # everything matching at least one rule is kept.
    if filter_keep:
        queries = []
        for filter_spec in filter_keep.values():
            q = spec_to_query_string(filter_spec)
            queries.append(f"({q})")
        query = " | ".join(queries)
        data.query(query, inplace=True)

    # Filters for removing data are negated and combined with "and" so that
    # only rows which don't match any rule are kept.
    if filter_remove:
        queries = []
        for filter_spec in filter_remove.values():
            q = spec_to_query_string(filter_spec)
            queries.append(f"~({q})")
        query = " & ".join(queries)
        data.query(query, inplace=True)

    data.reset_index(drop=True, inplace=True)


def fill_from_other_col(
    df: pd.DataFrame,
    *,
    coords_value_filling: dict[str, dict[str, dict[str, str]]],
    attrs: dict[str, Any],
) -> pd.DataFrame:
    """
    This function fills value in one column based on values in other columns.
    It can be used to fill NaN values or to replace e.g. non-standard or
    non-unique category codes based on category names. It operates on pandas
    DataFrames.


    Parameters
    ----------
    df : pd.DataFrame
        Data to operate on

    coords_value_filling : dict
        A dict with primap2 dimension names as keys. These are the target columns where
        values will be filled (or replaced). Values are dicts with primap2 dimension
        names as keys. These are the source columns. The values are dicts with source
        value - target value mappings.
        This can be used to e.g. fill missing category codes based on category names or
        to replace category codes which do not meet the terminology using the category
        names.

    attrs : dict
        Dataset attributes

    Returns
    -------
    pd.DataFrame
    """
    dim_aliases = _selection.translations_from_attrs(attrs, include_entity=True)

    # loop over target columns in value mapping
    for target_col in coords_value_filling:
        target_info = coords_value_filling[target_col]
        # loop over source columns
        for source_col in target_info:
            mapping_info = target_info[source_col]
            # loop over cases
            target_col_name = dim_aliases.get(target_col, target_col)
            source_col_name = dim_aliases.get(source_col, source_col)
            for source_value in mapping_info:
                df.loc[df[source_col_name] == source_value, target_col_name] = df.loc[
                    df[source_col_name] == source_value, target_col_name
                ] = mapping_info[source_value]
    return df


def add_dimensions_from_defaults(
    data: pd.DataFrame,
    coords_defaults: dict[str, Any],
    additional_allowed_coords: Iterable[str] = (),
):
    if_columns = (
        INTERCHANGE_FORMAT_OPTIONAL_COLUMNS
        + INTERCHANGE_FORMAT_MANDATORY_COLUMNS
        + list(additional_allowed_coords)
    )
    for coord in coords_defaults.keys():
        if coord in if_columns or coord.startswith(SEC_CATS_PREFIX):
            # add column to dataframe with default value
            data[coord] = coords_defaults[coord]
        else:
            raise ValueError(
                f"{coord!r} given in coords_defaults is unknown - prefix with "
                f"{SEC_CATS_PREFIX!r} to add a secondary category."
            )


def map_metadata(
    data: pd.DataFrame,
    *,
    meta_mapping: dict[str, str | Callable | dict],
    attrs: dict[str, Any],
):
    """Map the metadata according to specifications given in meta_mapping.
    First map entity, then the rest."""
    meta_mapping_copy = meta_mapping.copy()
    if "entity" in meta_mapping.keys():
        meta_mapping_entity = dict(entity=meta_mapping_copy["entity"])
        meta_mapping_copy.pop("entity")
        map_metadata_unordered(data, meta_mapping=meta_mapping_entity, attrs=attrs)

    map_metadata_unordered(data, meta_mapping=meta_mapping_copy, attrs=attrs)


def map_metadata_unordered(
    data: pd.DataFrame,
    *,
    meta_mapping: dict[str, str | Callable | dict],
    attrs: dict[str, Any],
):
    """Map the metadata according to specifications given in meta_mapping."""
    dim_aliases = _selection.translations_from_attrs(attrs, include_entity=True)

    # TODO: add additional mapping functions here
    # values: (function, additional arguments)
    mapping_functions = {
        "PRIMAP1": {
            "category": (_conversion.convert_ipcc_code_primap_to_primap2, []),
            "entity": (_conversion.convert_entity_gwp_primap_to_primap2, []),
            "unit": (
                _conversion.convert_unit_to_primap2,
                [dim_aliases.get("entity", "entity")],
            ),
        }
    }

    meta_mapping_df = {}
    # preprocess meta_mapping
    for column, mapping in meta_mapping.items():
        column_name = dim_aliases.get(column, column)
        if isinstance(mapping, str) or callable(mapping):
            if isinstance(mapping, str):  # need to translate to function first
                try:
                    func, args = mapping_functions[mapping][column]
                except KeyError:
                    logger.error(
                        f"Unknown metadata mapping {mapping!r} for column {column!r}, "
                        f"known mappings are: {list(mapping_functions.keys())}."
                    )
                    raise ValueError(
                        f"Unknown metadata mapping {mapping!r} for column {column!r}."
                    ) from None
            else:
                func = mapping
                args = []

            if not args:  # simple case: no additional args needed
                values_to_map = data[column_name].unique()
                values_mapped = map(func, values_to_map)
                meta_mapping_df[column_name] = dict(
                    zip(values_to_map, values_mapped, strict=False)
                )

            else:  # need to supply additional arguments
                # this can't be handled using the replace()-call later since the
                # mapped values don't depend on the original values only, therefore
                # we do it directly
                sel = [column_name, *args]
                values_to_map = np.unique(data[sel].to_records(index=False))
                for vals_to_map in values_to_map:
                    # we replace values where all the arguments match - build a
                    # selector for that, then do the replacement
                    selector = data[column_name] == vals_to_map[0]
                    for i, arg in enumerate(args):
                        selector &= data[arg] == vals_to_map[i + 1]

                    data.loc[selector, column_name] = func(*vals_to_map)

        else:
            meta_mapping_df[column_name] = mapping

    data.replace(meta_mapping_df, inplace=True)


def rename_columns(
    data: pd.DataFrame,
    coords_cols: dict[str, str],
    add_coords_cols: dict[str, list[str]],
    coords_defaults: dict[str, Any],
    coords_terminologies: dict[str, str],
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
            if coord == "entity":
                attrs["entity_terminology"] = coords_terminologies[coord]
        else:
            name = coord

        if coord.startswith(SEC_CATS_PREFIX):
            name = name[len(SEC_CATS_PREFIX) :]
            sec_cats.append(name)
        elif coord in attr_names:
            attrs[attr_names[coord]] = name

        coord_renaming[coords_cols.get(coord, coord)] = name

    for coord in add_coords_cols:
        coord_renaming[add_coords_cols[coord][0]] = coord

    data.rename(columns=coord_renaming, inplace=True)

    if sec_cats:
        attrs["sec_cats"] = sec_cats

    return attrs


_special_codes = {
    "C": np.nan,
    "nan": np.nan,
    "NaN": np.nan,
    "-": 0,
    "NE0": np.nan,
    "": np.nan,
}


def is_float(to_test: Any) -> bool:
    try:
        float(to_test)
        return True
    except ValueError:
        return False


def find_str_values_in_data(
    data: pd.DataFrame,
    columns: list[str],
) -> list:
    """Find all string values occurring in given columns of a DataFrame"""
    # limit our analysis to columns that contain strings
    # (or other object types)
    cols_with_strs = (
        data[columns].select_dtypes(include=[object]).columns.values.tolist()
    )
    temp = []
    for col in cols_with_strs:
        temp += list(data[col].unique())
    temp = list(set(temp))
    strs = [x for x in temp if not is_float(x)]
    return strs


def parse_code(code: str) -> float:
    """Parse a string code and return 0 or np.nan based on rules to interpret
    the codes."""
    code = code.strip()
    if code in _special_codes:
        return _special_codes[code]

    parts = code.split(",")
    parts = [x.replace(".", "").strip().upper() for x in parts]
    if "IE" in parts or "NO" in parts:
        return 0
    if "NE" in parts or "NA" in parts:
        return np.nan
    raise ValueError(f"Could not parse code: {code!r}.")


def create_str_replacement_dict(
    strs: list[str],
    user_str_conv: bool | dict[str, float],
) -> dict[str, str]:
    """Create a dict for replacement of strings by NaN and 0 based on
    general rules and user defined rules"""

    if isinstance(user_str_conv, bool):
        if user_str_conv:
            user_str_conv = {}
    elif isinstance(user_str_conv, dict):
        pass
    else:
        raise ValueError(
            f"Input for user_str_conv to create_str_replacement_dict"
            f" has to be a bool or a dict. {user_str_conv} is neither."
        )

    mapping = {}
    for str_val in strs:
        if str_val in user_str_conv:
            mapping[str_val] = user_str_conv[str_val]
        else:
            mapping[str_val] = parse_code(str_val)

    return mapping


def replace_values(data: pd.DataFrame, columns: list[str], na_repl_dict):
    """Replace str values indicating not-a-number by float NaN."""
    for col in columns:
        data[col] = data[col].replace(na_repl_dict)
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].astype("float64", copy=False, errors="ignore")


def preferred_unit(entity: str, units: dict[str, str]) -> str | None:
    """Choose the preferred unit for the given entity.

    In general, "Gg <substance> / year" will be preferred if it is compatible with the
    given input units. Otherwise, the first unit from units will be preferred.

    Parameters
    ----------
    entity: str
        Basic entity, e.g. a gas.
    units: dict of str, str
        Keys are the units which are in use for the entity and values the global
        warming potential specifications which will be used for the conversion.
        By specifying the gwp_context to use, you can make sure that it will be possible to
        convert from the input units to the output unit with the given gwp.
        If not applicable for a unit use None

    Returns
    -------
    preferred_unit: str
        The best unit for the given entity and units.

    Examples
    --------
    >>> preferred_unit("CO2", {"kt CO2 / yr": None, "mg CO2 / s": None})
    'Gg CO2 / yr'
    >>> preferred_unit("CH4", {"kt CO2 / yr": "AR4GWP100"})
    'Gg CH4 / yr'
    >>> preferred_unit("CH4", {"kt CO2 / yr": None, "Mg CH4 / yr": None})
    None
    >>> preferred_unit("CH4", {"kt CO2 / yr": "AR4GWP100", "Gg CO2 / yr": "SARGWP100"})
    'Gg CH4 / yr'
    >>> preferred_unit(
    ...     "KYOTOGHG", {"kt CO2 / yr": "AR4GWP100", "Gg CO2 / yr": "SARGWP100"}
    ... )

    """
    unit_fallback = next(iter(units.keys()))
    context_fallback = units[unit_fallback]
    # check if all can be converted to native or fallback units
    native_conv = []
    fb_conv = []
    native_unit = "Gg " + entity + " / yr"
    for unit in units.keys():
        conversion_contexts = []
        if units[unit] is not None:
            conversion_contexts.append(units[unit])

        # check if conversion to native unit is possible
        try:
            # print(f"Testing conversion from {ureg[unit_fallback].units} to "
            #       f"{ureg[native_unit].units} for {entity}.")
            if ureg(unit).is_compatible_with(ureg[native_unit], *conversion_contexts):
                native_conv.append(True)
            else:
                native_conv.append(False)
        except pint.UndefinedUnitError:
            # we have a gas basket or something unknown, so no conversion to native unit
            # print(f"Exception occurred for entity {entity}")
            native_conv.append(False)
            pass

        # check if conversion to fallback unit is possible
        if units[unit] != context_fallback:
            fb_conv.append(False)
            pass
        else:
            try:
                # print(f"Testing conversion from {ureg[unit_fallback].units} to "
                #       f"{ureg[native_unit].units} for {entity}.")
                if ureg(unit).is_compatible_with(
                    ureg[unit_fallback], *conversion_contexts
                ):
                    fb_conv.append(True)
                else:
                    fb_conv.append(False)
            except pint.UndefinedUnitError:
                # we have a gas basket or something unknown, so no conversion to native
                # unit
                # print(f"Exception occurred for entity {entity}")
                fb_conv.append(False)
                pass

    if all(native_conv):
        # print(f"converting {entity} to native unit {native_unit}")
        return native_unit
    elif all(fb_conv):
        # print(f"converting {entity} to fallback unit {unit_fallback}")
        return unit_fallback
    else:
        return None


def harmonize_units(
    data: pd.DataFrame,
    *,
    unit_col: None | str = None,
    attrs: None | dict = None,
    dimensions: Iterable[str],
) -> None:
    """Harmonize the units of the input data.

    For each entity, convert all time series to the same unit (the native unit or
    the unit that occurs first if conversion to the native unit is not possible). Units
    must already be in PRIMAP2 style.

    As unit handling is tricky and with new units new problem occur this function has a
    lot of (currently commented) debug output

    Parameters
    ----------
    data: pd.DataFrame
        data for which the units should be harmonized
    unit_col: str, optional
        column name for unit column. Default: "unit"
    attrs: dict, optional
        attrs defining the aliasing of columns. If attrs contains "entity_terminology",
        "entity (<entity_terminology>)" will be used as the entity column, otherwise
        simply "entity" will be used as the entity column.
    dimensions: list of str
        the dimensions, i.e. the metadata columns.

    Returns
    -------
    None
        The data is altered in place.
    """
    # we need to convert the data such that we have one unit per entity
    data_cols = list(set(data.columns.values) - set(dimensions))

    if attrs is not None:
        dim_aliases = _selection.translations_from_attrs(attrs, include_entity=True)
        entity_col = dim_aliases.get("entity", "entity")
    else:
        entity_col = "entity"

    if unit_col is None:
        unit_col = dim_aliases.get("unit", "unit")

    # find basic entities for all entities and make a list
    entities = data[entity_col].unique()
    basic_entities = {}

    # print(entities)
    for entity in entities:
        # check if GWP given in entity
        # print(f"entity: {entity}")
        gwp_match = re.findall(r"\(([A-Z0-9]*)\)$", entity)
        if gwp_match:
            gwp_to_use = gwp_match[0]
            basic_entity = re.findall(r"^[^\(\)\s]*", entity)
            basic_entity = basic_entity[0]
        else:
            gwp_to_use = None
            basic_entity = entity
        # print(f"gwp: {gwp_to_use}")
        # print(f"basic_entity: {basic_entity}")
        if basic_entity in basic_entities.keys():
            basic_entities[basic_entity][entity] = gwp_to_use
        else:
            basic_entities[basic_entity] = {entity: gwp_to_use}

    for basic_entity in basic_entities:
        # print(f"basic_entity: {basic_entity}")
        # print(f"entities: {basic_entities[basic_entity]}")
        # get all units for this entity
        data_this_basic_entity = data.loc[
            data[entity_col].isin(basic_entities[basic_entity])
        ]
        units_this_basic_entity = data_this_basic_entity[unit_col].unique()
        unit_gwp_this_basic_entity = {}
        gwp_conversion_this_basic_entity = False
        for entity in basic_entities[basic_entity].keys():
            data_this_entity = data_this_basic_entity.loc[data[entity_col] == entity]
            units_this_entity = data_this_entity[unit_col].unique()
            for unit in units_this_entity:
                unit_gwp_this_basic_entity[unit] = basic_entities[basic_entity][entity]
            if basic_entities[basic_entity][entity] is not None:
                gwp_conversion_this_basic_entity = True

            if len(units_this_basic_entity) > 1 or gwp_conversion_this_basic_entity:
                # need unit conversion.
                # determine unit to convert all units to. If none is found no conversion
                # is carried out at all
                unit_to = preferred_unit(basic_entity, unit_gwp_this_basic_entity)
                # print(f"basic_entity: {basic_entity}, unit_to: {unit_to}")
                if unit_to is not None:
                    # print(f"unit conversion for {basic_entity}, "
                    #      f"{basic_entities[basic_entity]}")
                    for entity in basic_entities[basic_entity]:
                        data_this_entity = data.loc[data[entity_col] == entity]
                        units_this_entity = data_this_entity[unit_col].unique()

                        for unit in units_this_entity:
                            if unit != unit_to:
                                # print(f"Working on unit {unit}")
                                unit_pint = ureg[unit]
                                # could add a try except block here to throw and log an
                                # error or add error info in DF instead of crashing
                                gwp_this_entity = basic_entities[basic_entity][entity]
                                if gwp_this_entity:
                                    with ureg.context(gwp_this_entity):
                                        unit_pint = unit_pint.to(unit_to)
                                else:
                                    unit_pint = unit_pint.to(unit_to)
                                # print(f"Pint unit is {unit_pint}")
                                factor = unit_pint.magnitude
                                # print(f"Converting with factor {factor} to unit
                                # {unit_to}")
                                mask = (data[entity_col] == entity) & (
                                    data[unit_col] == unit
                                )
                                # print(data.loc[mask, data_cols])
                                try:
                                    data.loc[mask, data_cols] *= factor
                                except TypeError:
                                    # print(data.loc[mask, data_cols])
                                    strs = find_str_values_in_data(data, data_cols)
                                    logger.error(
                                        f"The following string values are present and "
                                        f"can not be converted during unit conversion: "
                                        f"{strs}."
                                    )
                                    raise ValueError(
                                        f"String values {strs} prevent unit conversion."
                                    ) from None

                                data.loc[mask, unit_col] = unit_to

                        # if entity differs from basic entity and the units are not
                        # compatible we had GWP conversion and have to adapt the entity
                        if (entity != basic_entity) and not ureg(
                            unit
                        ).is_compatible_with(ureg[unit_to]):
                            # entity was converted
                            entity_mask = data[entity_col] == entity
                            # print(f"Changing entity from {entity} to {basic_entity}")
                            data.loc[entity_mask, entity_col] = basic_entity


def sort_columns_and_rows(
    data: pd.DataFrame,
    dimensions: Iterable[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Sort the data.

    The columns are ordered according to the order in
    INTERCHANGE_FORMAT_COLUMN_ORDER, with secondary categories alphabetically after
    the category and all date columns in order at the end.

    The rows are ordered by values of the non-date columns.

    Parameters
    ----------
    data: pd.DataFrame
        data which should be ordered
    dimensions: list of str
        the dimensions, i.e. the metadata columns.

    Returns
    -------
    sorted, dimensions_sorted : (pd.DataFrame, list of str)
        the input data frame with columns and rows ordered and the dimensions sorted.
    """
    time_cols = list(set(data.columns.values) - set(dimensions))

    other_cols = list(dimensions)
    cols_sorted = []
    for col in INTERCHANGE_FORMAT_COLUMN_ORDER:
        for ocol in other_cols:
            if ocol == col or (isinstance(ocol, str) and ocol.startswith(f"{col} (")):
                cols_sorted.append(ocol)
                other_cols.remove(ocol)
                break

    cols_sorted += list(sorted(other_cols))

    data: pd.DataFrame = data[cols_sorted + list(sorted(time_cols))]

    data.sort_values(by=cols_sorted, inplace=True)
    data.reset_index(inplace=True, drop=True)

    return data, cols_sorted
