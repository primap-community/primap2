import itertools
import os
import re

import pandas as pd
import xarray as xr
from loguru import logger

from primap2._units import ureg


def convert_dataframe_units_primap_to_primap2(
    data_frame: pd.DataFrame, unit_col: str = "unit", entity_col: str = "entity"
):
    """
    This function converts the emissions module style units which usually neither carry
    information about the substance nor about the time to primap2 units. The function
    also handles the exception cases where PRIMAP units do contain information about the
    substance (e.g. GtC).

    The function operates in a ScmRun dataframe which is alteed in place.

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
        data_frame[unit_col].astype(str) + " " + data_frame[entity_col].astype(str)
    )

    # get unique entity_unit combinations
    unique_entity_unit = data_frame["entity_unit"].unique()
    for entity_unit in unique_entity_unit:
        [cur_entity, cur_unit] = re.split(" ", entity_unit)
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

    # define exceptions
    # specialities
    exception_units = {
        "CO2eq": "CO2",  # convert to just CO2
        "<entity>N": "N",
        "C": "C",  # don't add variable here (untested)
    }

    # basic units
    basic_units = ["g", "t"]

    # prefixes
    si_unit_multipliers = ["k", "M", "G", "T", "P", "E", "Z", "Y"]

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
        match = re.search(regexp_str, unit_entity)
        if match is None:
            logger.error("No unit prefix matched for unit." + unit_entity)
            raise RuntimeError("No unit prefix matched for unit." + unit_entity)
        pref_basic = match.group(0)
        # then get variable
        match = re.search("(?<=\s)(.*)(?=\s/\s)", unit_entity)  # noqa: W605
        if match is None:
            logger.error("No variable matched for unit." + unit_entity)
            raise RuntimeError("No unit variable matched for unit." + unit_entity)
        entity = match.group(0)
        # now build the replacement
        converted_unit = (
            pref_basic + " " + exception_units[exception_unit] + time_frame_str
        )
        # add the variable if necessary
        converted_unit = converted_unit.replace("<entity>", entity)

    else:
        # standard unit
        converted_unit = unit_entity

    return converted_unit


def dates_to_dimension(ds: xr.Dataset, time_format: str = "%Y") -> xr.DataArray:
    """
    This function converts a xr dataset where each time point is an individual
    coordinate to a xr.DataArray with one time coordinate and the time points as values.
    All dimensions which are not in the index are assumed to be time points.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with individual time points as dimensions

    time_format : str
        format string for the time points. Default: %Y (for year only)

    Returns
    -------
    :obj:`xr.DataArray`
        xr.DataArray with the time as an dimension and time points as values

    """
    ds = ds.to_array("time").unstack().dropna("time", "all")
    ds["time"] = pd.to_datetime(ds["time"].values, format="%Y", exact=False)
    return ds


def metadata_for_entity_primap(unit: str, entity: str) -> dict:
    """
    This function takes GWP information from the entity name in primap style (if
    present). Information is returned as attrs dict.

    Currently the function uses a limited set of GWP values (define in gwp_mapping) and
    works on a limited set of variables (defined in entities_gwp).

    Parameters
    ----------
    unit: str
        unit to be stored in the attrs dict
    entity: str
        entity to process

    Returns
    -------
    :dict:
    dict containing new variable name and attrs for use in xr Dataset variable

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
    found = re.match(regexp_str, entity)
    if found is None:
        # not a basket entity which uses GWPs
        entity_out = entity
        variable_name = entity
        attr_dict = {"entity": entity_out, "unit": unit}
    else:
        # check if GWP information present in entity
        match = re.search(regexp_str_gwps, entity)
        if match is None:
            # SAR GWPs are default in PRIMAP
            gwp_out = gwp_mapping["SAR"]
            entity_out = entity
        else:
            gwp_out = match.group(0)
            # in this case the entity has to be replaced as well
            match = re.search(".*(?=" + gwp_out + ")", entity)
            if match.group() == "":
                # TODO: throw error this should be impossible
                print("Error in GWP processing")
                return
            else:
                gwp_out = gwp_mapping[gwp_out]
                entity_out = match.group(0)
        variable_name = entity_out + " (" + gwp_out + ")"
        attr_dict = {"entity": entity_out, "unit": unit, "gwp_context": gwp_out}

    result_dict = {"variable_name": variable_name, "attrs": attr_dict}

    return result_dict


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

    input:
        code = 'IPC1A3B34'

    output:
        '1.A.3.b.3.iv'

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
        "9": "viiii",
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

    # TODO: checks at all levels (currently only a few)

    if len(code) < 4:
        # code too short
        # TODO: throw error
        print("Category code " + code + " is too short. Not converted")
    elif code[0:3] in ["IPC", "CAT"]:
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

            # first two chars are unchanged but a dot is added
            if len(code_remaining) == 1:
                new_code = code_remaining
            elif len(code_remaining) == 2:
                new_code = code_remaining[0] + "." + code_remaining[1]
            else:
                new_code = code_remaining[0] + "." + code_remaining[1]
                code_remaining = code_remaining[2:]
                # the next part is a number. match by regexp to also match 2 digit
                # numbers (just in case there are any, currently not needed)
                match = re.match("[0-9]*", code_remaining)
                if match is None:
                    # code does not obey specifications. Throw a warning and stop
                    # conversion
                    # TODO: throw warning
                    print(
                        "Category code "
                        + code
                        + " does not obey spcifications. No number found on third level"
                    )
                    new_code = ""
                else:
                    new_code = new_code + "." + match.group(0)
                    code_remaining = code_remaining[len(match.group(0)) :]

                    # fourth level is a char. Has to be transformed to lower case
                    if len(code_remaining) > 0:
                        new_code = new_code + "." + code_remaining[0].lower()
                        code_remaining = code_remaining[1:]

                        # now we have an arabic numeral in the PRIMAP-format but a roman
                        # numeral in pyCPA
                        if len(code_remaining) > 0:
                            new_code = (
                                new_code + "." + arabic_to_roman[code_remaining[0]]
                            )
                            code_remaining = code_remaining[1:]

                            if len(code_remaining) > 0:
                                # now we have a number again. An it's the end of the
                                # code. So just copy the rest
                                new_code = new_code + "." + code_remaining

        return new_code

    else:
        # the prefix is missing. Throw a warning and don't do anything
        # TODO: throw warning
        print(
            "Category code "
            + code
            + " is not in PRIMAP IPCC code format. Not converted"
        )
        return ""


def read_wide_csv_file(
    file: str,
    folder: str,
    coords_cols: dict = {},
    coords_defaults: dict = {},
    coords_terminologies: dict = {},
    meta_mapping: dict = {},
    filter_keep: dict = {},
    filter_remove: dict = {},
) -> xr.Dataset:
    """
    This function reads a csv file and returns the data as an ScmDataFrame

    Columns can be renamed or filled with default vales to match the pyCPA structure.
    The individual files are read using "read_wide_csv_data"

    Currently duplicate datapoint will not be detected
    TODO: so far only one secondary category can be set
    TODO: enable filtering through query strings

    Parameters
    ----------

    file : str
        String containing the file name

    folder : str
        String with folder name to read from

    coords_cols : dict
        Dict where the keys are column names in the files to be read and the value is
        the column file in the output dataframe. Default = empty

    coords_defaults : dict
        Dict for default values of columns not given in the csv files. The keys are the
        column names and the values are the values for the columns. Default: empty

    coords_terminologies : dict
        Dict defining the terminologies used for the different coordinates (e.g. ISO3
        for area) Default: empty

    meta_mapping : dict
        A dict with primap2 column names as keys. Values are dicts with input values as
        keys and output values as values. A standard use case is to map gas names from
        input data to the standardized names used in primap2.
        Alternatively values can be strings which define the function to run to create
        the dict. Possible functions are hard-coded currently for security reasons.
        Default: empty

    filter_keep : dict
        Dict defining filters of data to keep. Filtering is done after metadata mapping,
        so use mapped metadata values to define the filter. Column names are primap2
        columns. Each entry in the dict defines an individual filter. The names of the
        filters have no relevance. Default: empty

    filter_remove : dict
        Dict defining filters of data to remove. Filtering is done after metadata
        mapping, so use mapped metadata values to define the filter. Column names are
        primap2 columns. Each entry in the dict defines an individual filter. The names
        of the filters have no relevance. Default: empty

    Returns
    -------

    :obj:`xr.Dataset`
        xr dataset with the read data


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

    # TODO: check if mandatory coords present in input definitions
    #  (do we have the list of mandatory coords somewhere)

    # 1) read the data
    # read the data into a pandas dataframe replacing special strings by nan
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
    read_data = pd.read_csv(os.path.join(folder, file), na_values=na_values)

    # 2) prepare and filter the dataset
    # get all the columns that are actual data not metadata (usually the years)
    year_regex = re.compile("\d")  # noqa: W605
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

    # filter the data
    # first the filters for keeping data. Filters are combined with "or" while all
    # column conditions in each filter are combined with "&"
    if filter_keep:
        query = ""
        for filter_name in filter_keep:
            filter_current = filter_keep[filter_name]
            query = query + "("
            for col in filter_current:
                if isinstance(filter_current[col], list):
                    query = query + col + " in ["
                    for item in filter_current[col]:
                        query = query + "'" + item + "', "
                    query = query[0:-2] + "] & "
                else:
                    query = query + col + " == '" + filter_current[col] + "' & "
            query = query[0:-2] + ") |"
        query = query[0:-2]
        # print("filter_keep: " + query)
        read_data.query(query, inplace=True)

    # now the filters for removing data. Same as for keeping
    if filter_remove:
        query = "~"
        for filter_name in filter_remove:
            filter_current = filter_remove[filter_name]
            query = query + "("
            for col in filter_current:
                if isinstance(filter_current[col], list):
                    query = query + col + " in ["
                    for item in filter_current[col]:
                        query = query + "'" + item + "', "
                    query = query[0:-2] + "] & "
                else:
                    query = query + col + " == '" + filter_current[col] + "' & "
            query = query[0:-2] + ") |"
        query = query[0:-2]
        # print("filter_remove: " + query)
        read_data.query(query, inplace=True)

    # 3) bring metadata into primap2 format (mandatory metadata, entity names, unit etc)
    # map metadata (e.g. replace gas names, scenario names etc.)
    # TODO: add conversion functions here!
    if meta_mapping:
        # preprocess meta_mapping
        for column in meta_mapping:
            if isinstance(meta_mapping[column], str):
                if column == "category":
                    # print('creating meta_mapping dict for ' + column)
                    values_to_map = read_data[coords_cols[column]].unique()
                    if meta_mapping[column] == "PRIMAP":
                        # print("using PRIMAP converter")
                        values_mapped = map(
                            convert_ipcc_code_primap_to_primap2, values_to_map
                        )
                    else:
                        # TODO: throw error
                        return
                else:
                    # TODO: throw error
                    return
                meta_mapping[column] = dict(zip(values_to_map, values_mapped))
                # print(meta_mapping[column])
        read_data.replace(meta_mapping, inplace=True)

    # commented as not needed if we create the xarray directly. But needed if we want to
    # use a pandas interchange format as intermediate step
    # mandatory_columns = ["area", "source"]
    # # if entity col is not in data add with default value
    # for column in mandatory_columns:
    #     if column in coords_cols.keys():
    #         pass
    #     elif column in coords_defaults.keys():
    #         read_data[column] = coords_defaults[column]
    #     else:
    #         # TODO throw error
    #         print("error")

    # entity is also mandatory as it is transformed into the variables
    if "entity" in coords_cols.keys():
        entity_col = coords_cols["entity"]
    elif "entity" in coords_defaults.keys():
        entity_col = "entity"
        read_data[entity_col] = coords_defaults["entity"]
    else:
        # TODO throw error
        print("error")

    # if we have a unit col, we need to convert the units to primap2 (pint) style and
    # then convert the data such that we have one unit per entity
    # TODO: this could be moved into an individual function (e.g. harmonize_units)
    if "unit" in coords_cols.keys():
        unit_col = coords_cols["unit"]
        # first convert the units such that pint understands them
        # TODO: this needs to be flexible as there are different possible input formats
        convert_dataframe_units_primap_to_primap2(
            read_data, unit_col=unit_col, entity_col=entity_col
        )
        # get entities from read data
        entities = read_data[entity_col].unique()
        for entity in entities:
            # get all units for this entity
            data_this_entity = read_data.loc[read_data[entity_col] == entity]
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
                    read_data.loc[
                        (read_data[entity_col] == entity)
                        & (read_data[unit_col] == unit),
                        year_cols,
                    ] = (
                        read_data.loc[
                            (read_data[entity_col] == entity)
                            & (read_data[unit_col] == unit),
                            year_cols,
                        ]
                        * factor
                    )
                    read_data.loc[
                        (read_data[entity_col] == entity)
                        & (read_data[unit_col] == unit),
                        unit_col,
                    ] = unit_to
    elif "unit" in coords_defaults.keys():
        # set unit. no unit conversion needed
        # for interchange format create unit col here
        pass
    else:
        # no unit information present. Throw an error
        # TODO: throw error
        print("error")

    # print(read_data.columns.values)
    # 4) convert to primap2 xarray format
    read_data_xr = read_data.to_xarray()
    index_cols = list(set(columns) - set(year_cols) - set([unit_col]))
    # print(index_cols)
    read_data_xr = read_data_xr.set_index({"index": index_cols})
    # take the units out as they increase dimensionality and we have only one unit per
    # entity/variable
    units = read_data_xr[unit_col].dropna("index")
    del read_data_xr[unit_col]
    # print(read_data_xr.index)

    # the next step is very memory consuming if the dimensionality of the array is too
    # high
    # TODO: introduce a check of dimensionality and alternate method for large datasets
    read_data_xr = dates_to_dimension(read_data_xr).to_dataset("entity")

    # fill the entity/variable attributes
    # TODO: it would be better if the renaming is done earlier such that the exchange
    #  format data is complete before conversion
    # to xarray
    entity_renaming = {}
    for entity in read_data_xr:
        csv_units = units.loc[{"entity": entity}].to_series().unique()
        if len(csv_units) > 1:
            raise ValueError(
                f"More than one unit for {entity!r}: {csv_units!r}. "
                f"There is an error in the unit conversion."
            )
        mass_unit = csv_units[0]
        entity_info = metadata_for_entity_primap(mass_unit, entity)
        read_data_xr[entity].attrs = entity_info["attrs"]
        entity_renaming[entity] = entity_info["variable_name"]

    read_data_xr = read_data_xr.rename(entity_renaming)

    # add the dataset wide attributes and rename dimensions accordingly
    attrs = {}
    coord_renaming = {}
    for coord in coords_cols:
        if coord in coords_terminologies.keys():
            name = coord + " (" + coords_terminologies[coord] + ")"
        else:
            name = coord
        attrs[coord] = name
        if coords_cols[coord] != name:
            coord_renaming[coords_cols[coord]] = name

    for coord in coords_defaults:
        if coord in read_data_xr.dims:
            if coord in coords_terminologies.keys():
                name = coord + " (" + coords_terminologies[coord] + ")"
            else:
                name = coord
            attrs[coord] = name
            if coords_defaults[coord] != name:
                coord_renaming[coords_defaults[coord]] = name
        else:
            attrs[coord] = coords_defaults[coord]

    remove_cols = ["unit", "entity"]
    for col in remove_cols:
        if col in coord_renaming:
            del coord_renaming[col]
        if col in attrs:
            del attrs[col]
    read_data_xr = read_data_xr.rename_dims(coord_renaming)
    read_data_xr = read_data_xr.rename(coord_renaming)
    read_data_xr.attrs = attrs
    read_data_xr = read_data_xr.pr.quantify()

    return read_data_xr
