"""Tests for _data_reading.py"""
# import logging

# import pandas as pd
# import pint
import pytest

import primap2.io._data_reading as pm2io

# import xarray as xr


# @pytest.mark.parametrize("unit_in, entity_in, expected_unit_out", [("3+5", 8),
# ("2+4", 6), ("6*9", 42)])
# def test_convert_unit_primap_to_primap2():
#    assert convert_unit_primap_to_primap2(
#        data_frame: pd.DataFrame,
#    unit_col: str = "unit",
#                    entity_col: str = "entity"


@pytest.mark.parametrize(
    "code_in, expected_code_out",
    [("IPC1A", "1.A"), ("CATM0EL", "M.0.EL"), ("IPC1A1B23", "1.A.1.b.ii.3")],
)
def test_convert_ipcc_code_primap_to_primap2(code_in, expected_code_out):
    assert pm2io.convert_ipcc_code_primap_to_primap2(code_in) == expected_code_out


@pytest.mark.parametrize(
    "unit_in, entity_in, expected_unit_out",
    [
        ("GgCO2eq", "KYOTOGHG", "Gg CO2 / yr"),
        ("MtC", "CO", "Mt C / yr"),
        ("GgN2ON", "N2O", "Gg N / yr"),
        ("t", "CH4", "t CH4 / yr"),
    ],
)
def test_convert_unit_primap_to_primap2(unit_in, entity_in, expected_unit_out):
    assert pm2io.convert_unit_primap_to_primap2(unit_in, entity_in) == expected_unit_out


@pytest.mark.parametrize(
    "unit, entity, expected_dict",
    [
        (
            "Mt",
            "CO2",
            {"attrs": {"unit": "Mt", "entity": "CO2"}, "variable_name": "CO2"},
        ),
        (
            "Gg CO2",
            "KYOTOGHGAR4",
            {
                "attrs": {
                    "unit": "Gg CO2",
                    "entity": "KYOTOGHG",
                    "gwp_context": "AR4GWP100",
                },
                "variable_name": "KYOTOGHG (AR4GWP100)",
            },
        ),
        (
            "kg CO2",
            "KYOTOGHG",
            {
                "attrs": {
                    "unit": "kg CO2",
                    "entity": "KYOTOGHG",
                    "gwp_context": "SARGWP100",
                },
                "variable_name": "KYOTOGHG (SARGWP100)",
            },
        ),
    ],
)
def test_metadata_for_entity_primap(unit, entity, expected_dict):
    assert pm2io.metadata_for_entity_primap(unit, entity) == expected_dict
