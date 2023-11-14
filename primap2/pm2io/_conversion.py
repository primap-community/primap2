import itertools
import re

from loguru import logger

# basic units
_basic_units = ["g", "t"]

# prefixes
_si_unit_multipliers = ["", "k", "M", "G", "T", "P", "E", "Z", "Y"]

# combines basic units with prefixes
_units_prefixes = list(itertools.product(_si_unit_multipliers, _basic_units))
_units_prefixes = [i[0] + i[1] for i in _units_prefixes]

# build regexp to match the basic units with prefixes in units
_units_prefixes_regexp = "(" + "|".join(_units_prefixes) + ")"


def convert_unit_to_primap2(unit: str, entity: str) -> str:
    """Convert PRIMAP1 emissions module style units and units in similar formats to
    primap2 units.

    This function converts the emissions module style units which usually neither carry
    information about the substance nor about the time to primap2 units. The function
    also handles the exception cases where input units do contain information about the
    substance (e.g. GtC).

    Parameters
    ----------
    unit : str
        unit to convert
    entity : str
        entity for which the conversion takes place

    Returns
    -------
    unit : str
        converted unit
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
        "CO2e": "CO2",  # convert to just CO2 (not for PRIMAP but e.g. NIRs)
        "CO₂eq": "CO2",  # convert to just CO2 (not for PRIMAP but e.g. NIRs)
        "<entity>N": "N",
        "C": "C",  # don't add variable here
    }

    # time information to add
    time_frame_str = " / yr"

    # remove spaces for more flexibility in input units
    unit = unit.replace(" ", "")

    # check if entity contains GWP information. If so discard
    # not needed for PRIMAP1 entities but when using the function for data reading
    entity_match = re.match(r"^[^\(\)\s]*", entity)
    entity = entity_match[0]

    # add entity and time frame to unit
    # special units will be replaced later
    unit_entity = unit + " " + entity + time_frame_str

    # check if unit has prefix
    match_pref = re.search(_units_prefixes_regexp, unit_entity)
    if match_pref is None:
        logger.warning("No unit prefix matched for unit. " + unit_entity)
        return "error_" + unit + "_" + entity

    # check if exception unit
    is_ex_unit = [
        re.match(
            _units_prefixes_regexp + ex_unit.replace("<entity>", entity) + "$", unit
        )
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


def code_invalid_warn(code: str, message: str) -> str:
    """Log a warning and return an error code."""
    logger.warning(
        f"Category code {code!r} does not conform to specifications: {message}"
    )
    return "error_" + code


def convert_ipcc_code_primap_to_primap2(code: str) -> str:
    """Convert IPCC emissions category codes from PRIMAP1 emissions module style to
    primap2 style.

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
    code: str
        the category code in primap2 format

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
        "MLULUCF": "M.LULUCF",
        "MMULTIOP": "M.MULTIOP",
        "M0EL": "M.0.EL",
        "MBIO": "M.BIO",
        "M3B4APF": "M.3.B.4.APF",
        "M3B4APD": "M.3.B.4.APD",
        "M3CAG": "M.3.C.AG",
        "M3C1AG": "M.3.C.1.AG",
        "M3C1AGSAV": "M.3.C.1.AG.SAV",
        "M3C1AGRES": "M.3.C.1.AG.RES",
        "M3D2LU": "M.3.D.2.LU",
        "M.AG": "M.AG",
        "M.AG.ELV": "M.AG.ELV",
        "M.BK": "M.BK",
        "M.BK.A": "M.BK.A",
        "M.BK.M": "M.BK.M",
        "M.LULUCF": "M.LULUCF",
        "M.MULTIOP": "M.MULTIOP",
        "M.0.EL": "M.0.EL",
        "M.BIO": "M.BIO",
    }

    if code[0:3] not in ["IPC", "CAT"]:
        # prefix = ""
        pure_code = code
    elif len(code) < 4:
        return code_invalid_warn(
            code, "Too short to be a PRIMAP IPCC code after " + "removal of prefix."
        )
    else:
        # prefix = code[0:3]
        pure_code = code[3:]

    # it's an IPCC code. convert it
    # check if it's a custom code (beginning with 'M'). Those have to be either mapped
    # explicitly using "code_mapping" or follow the normal structure after the "M"

    # check if a separator between prefix and code is used
    if len(pure_code) == 0:
        return code_invalid_warn(
            code, "Pure code has length 0. This should not be possible."
        )
    if pure_code[0] in [".", " ", "_", "-"]:
        pure_code = pure_code[1:]

    if pure_code[0] == "M":
        code_remaining = pure_code
        if pure_code in code_mapping.keys():
            new_code = code_mapping[pure_code]
            return new_code
        else:
            new_code = "M."
            code_remaining = code_remaining[1:]
            if len(code_remaining) == 0:
                return code_invalid_warn(
                    code, "Nothing follows the 'M' for an 'M'-code."
                )
    else:
        new_code = ""
        # only work with the part without 'IPC' or 'CAT'
        code_remaining = pure_code

    # if the code ends with a dot remove it
    if code_remaining[-1] == ".":
        code_remaining = code_remaining[:-1]

    # whenever we encounter dots as separators between levels, we ignore them.
    if code_remaining[0] == ".":
        code_remaining = code_remaining[1:]

    # actual conversion happening here
    # first level is a digit
    if code_remaining[0].isdigit():
        new_code = new_code + code_remaining[0]
    else:
        return code_invalid_warn(code, "No digit found on first level.")

    # second level is a letter
    if len(code_remaining) > 1:
        code_remaining = code_remaining[1:]
        if code_remaining[0] == ".":
            code_remaining = code_remaining[1:]
        # no need to check if code_remaining is emprty as we ensured that
        # the last char is not a dot (same in the following steps)
        if code_remaining[0].isalpha():
            new_code = new_code + "." + code_remaining[0]
        else:
            return code_invalid_warn(code, "No letter found on second level.")

        # third level is a number. might be more than one char, so use regexp
        if len(code_remaining) > 1:
            code_remaining = code_remaining[1:]
            if code_remaining[0] == ".":
                code_remaining = code_remaining[1:]
            match = re.match("^[0-9]+", code_remaining)
            if match is not None:
                new_code = new_code + "." + match.group(0)
            else:
                return code_invalid_warn(code, "No number found on third level.")

            # fourth level is a letter. has to be transformed to lower case
            if len(code_remaining) > len(match.group(0)):
                code_remaining = code_remaining[len(match.group(0)) :]
                if code_remaining[0] == ".":
                    code_remaining = code_remaining[1:]
                if code_remaining[0].isalpha():
                    new_code = new_code + "." + code_remaining[0].lower()
                else:
                    return code_invalid_warn(code, "No letter found on fourth level.")

                # fifth level is digit in PRIMAP1 format but roman numeral in IPCC
                # and PRIMAP2
                if len(code_remaining) > 1:
                    code_remaining = code_remaining[1:]
                    if code_remaining[0] == ".":
                        code_remaining = code_remaining[1:]
                    if code_remaining[0].isdigit():
                        new_code = new_code + "." + arabic_to_roman[code_remaining[0]]
                        len_level_5 = 1
                    else:
                        # try to match a roman numeral
                        match = re.match("^[ivx]{1,4}", code_remaining)
                        if match is not None:
                            new_code = new_code + "." + match.group(0)
                            len_level_5 = len(match.group(0))
                        else:
                            return code_invalid_warn(
                                code, "No digit or roman numeral found on fifth level."
                            )

                    # sixth and last level is a number.
                    if len(code_remaining) > len_level_5:
                        code_remaining = code_remaining[len_level_5:]
                        if code_remaining[0] == ".":
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


def convert_entity_gwp_primap_to_primap2(entity_pm1: str) -> str:
    """Convert PRIMAP1 emissions module style entity names to primap2 style.

    The conversion only considers the GWP, currently the variable itself is
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
        "AR6": "AR6GWP100",
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
