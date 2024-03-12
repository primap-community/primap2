"""This file contains functions for reading of country GHG inventories
from National Inventory Reports (NIR), biannual Update Reports (BUR),
and other official country emissions inventories
Most of the functions in this file are exposed to the outside yet they
currently do not undergo the strict testing applied to the rest of PRIMAP2 as
they are added during the process of reading an preparing data for the PRIMAP-hist
update. Testing will be added in the future."""

import re

import pandas as pd


def nir_add_unit_information(
    df_nir: pd.DataFrame,
    *,
    unit_row: str | int,
    entity_row: str | int | None = None,
    regexp_entity: str,
    regexp_unit: str | None = None,
    manual_repl_unit: dict[str, str] | None = None,
    manual_repl_entity: dict[str, str] | None = None,
    default_unit: str,
) -> pd.DataFrame:
    """Add unit information to a National Inventory Report (NIR) style DataFrame.

    Add unit information to the header of an "entity-wide" file as
    present in the standard table format of National Inventory Reports (NIRs). The
    unit and entity information is extracted from combined unit and entity information
    in the row defined by `unit_row`. The parameters `regexp_unit` and `regexp_entity`
    determines how this is done by regular expressions for unit and entity.
    Additionally, manual mappings can be defined in the `manual_repl_unit` and
    `manual_repl_entity` dicts. For each column the routine tries to extract a unit
    using the regular expression. If this fails it looks in the `manual_repl_unit`
    dict for unit and in `manual_repl_entity` for entity information. If there is no
    information the default unit given in `default_unit` is used. In this case the
    analyzed value is used as entity unchanged.

    Parameters
    ----------
    df_nir : pd.DataFrame
        Pandas DataFrame with the table to process
    unit_row : str or int
        String "header" to indicate that the column header should be used to derive the
        unit information or an integer specifying the row to use for unit information.
        If entity and unit information are given in the same row use only unit_row.
    entity_row : str or int
        String "header" to indicate that the column header should be used to derive the
        unit information or an integer specifying the row to use for entity information.
        If entity and unit information are given in the same row use only unit_row
    regexp_entity : str
        regular expression that extracts the entity from the cell value
    regexp_unit : str (optional)
        regular expression that extracts the unit from the cell value
    manual_repl_unit : dict (optional)
        dict defining unit for given cell values
    manual_repl_entity : dict (optional)
        dict defining entity for given cell values
    default_unit : str
        unit to be used if no unit can be extracted an no unit is given

    Returns
    -------
    pd.DataFrame
        DataFrame with explicit unit information (as column header)
    """

    if manual_repl_unit is None:
        manual_repl_unit = {}

    if manual_repl_entity is None:
        manual_repl_entity = {}

    cols_to_drop = []

    # get the data to extract the units and entities from
    # can be either the header row or a regular row
    if unit_row == "header":
        values_for_units = list(df_nir.columns)
    else:
        # unit_row must be an integer
        values_for_units = list(df_nir.iloc[unit_row])
        cols_to_drop.append(unit_row)

    if entity_row is not None:
        if entity_row == "header":
            values_for_entities = list(df_nir.columns)
        else:
            values_for_entities = list(df_nir.iloc[entity_row])
            if entity_row != unit_row:
                cols_to_drop.append(entity_row)
    else:
        values_for_entities = values_for_units

    if regexp_unit is not None:
        re_unit = re.compile(regexp_unit)
    re_entity = re.compile(regexp_entity)

    units = values_for_units.copy()
    entities = values_for_entities.copy()

    for idx, value in enumerate(values_for_units):
        if str(value) in manual_repl_unit:
            units[idx] = manual_repl_unit[str(value)]
        else:
            if regexp_unit is not None:
                unit = re_unit.findall(str(value))
            else:
                unit = False

            if unit:
                units[idx] = unit[0]
            else:
                units[idx] = default_unit

    for idx, value in enumerate(values_for_entities):
        if str(value) in manual_repl_entity:
            entities[idx] = manual_repl_entity[str(value)]
        else:
            entity = re_entity.findall(str(value))
            if entity:
                entities[idx] = entity[0]
            else:
                entities[idx] = value

    new_header = [entities, units]

    df_out = df_nir.copy()
    df_out.columns = new_header
    if cols_to_drop:
        df_out = df_out.drop(df_out.index[cols_to_drop])

    return df_out


def nir_convert_df_to_long(
    df_nir: pd.DataFrame, year: int, header_long: list[str] | None = None
) -> pd.DataFrame:
    """Convert an entity-wide NIR table for a single year to a long format
    DataFrame.

    The input DataFrame is required to have the following structure:
    * Columns for category, original category name, and data in this order, where
    category and original category name form a multiindex.
    * Column header as multiindex for entity and unit
    A column for the year is added during the conversion.

    Parameters
    ----------
    df_nir: pd.DataFrame
        Pandas DataFrame with the NIR table to be converted
    year: int
        Year of the given data
    header_long: list, optional
        specify a non-standard column header, e.g. with only category code
        or orig_cat_name

    Returns
    -------
    pd.DataFrame
        converted DataFrame
    """
    if header_long is None:
        header_long = ["category", "orig_cat_name", "entity", "unit", "time", "data"]

    df_stacked = df_nir.stack([0, 1], dropna=True).to_frame()
    df_stacked.insert(0, "year", str(year))
    df_stacked = df_stacked.reset_index()
    df_stacked.columns = header_long
    return df_stacked
