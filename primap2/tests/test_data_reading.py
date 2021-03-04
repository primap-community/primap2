"""Tests for _data_reading.py"""
# import logging

import os

import pandas as pd

# import pint
import pytest

# import logging
import primap2 as pm2
import primap2.io._data_reading as pm2io

from .utils import assert_ds_aligned_equal


@pytest.mark.parametrize(
    "code_in, expected_code_out",
    [("IPC1A", "1.A"), ("CATM0EL", "M.0.EL"), ("IPC1A1B23", "1.A.1.b.ii.3")],
)
def test_convert_ipcc_code_primap_to_primap2(code_in, expected_code_out):
    assert pm2io.convert_ipcc_code_primap_to_primap2(code_in) == expected_code_out


def test_convert_ipcc_code_primap_to_primap2_too_short(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC") == "error_IPC"
    assert "WARNING" in caplog.text
    assert "Category code 'IPC' is too short to be a PRIMAP IPCC code." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_wrong_format(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPD1A") == "error_IPD1A"
    assert "WARNING" in caplog.text
    assert "Category code 'IPD1A' is not in PRIMAP IPCC code format." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_first_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPCA1") == "error_IPCA1"
    assert "WARNING" in caplog.text
    assert " No digit found on first level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_second_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC123") == "error_IPC123"
    assert "WARNING" in caplog.text
    assert " No letter found on second level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_third_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC1AC") == "error_IPC1AC"
    assert "WARNING" in caplog.text
    assert " No number found on third level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_fourth_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC1A2_") == "error_IPC1A2_"
    assert "WARNING" in caplog.text
    assert " No letter found on fourth level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_fifth_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC1A2BB") == "error_IPC1A2BB"
    assert "WARNING" in caplog.text
    assert " No digit found on fifth level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_sixth_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC1A2B3X") == "error_IPC1A2B3X"
    assert "WARNING" in caplog.text
    assert " No number found on sixth level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_after_sixth_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC1A2B33A") == "error_IPC1A2B33A"
    assert "WARNING" in caplog.text
    assert " Chars left after sixth level." in caplog.text


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


def test_convert_unit_primap_to_primap2_no_prefix(caplog):
    assert (
        pm2io.convert_unit_primap_to_primap2("CO2eq", "FGASES") == "error_CO2eq_FGASES"
    )
    assert "WARNING" in caplog.text
    assert "No unit prefix matched for unit." in caplog.text


def test_convert_unit_primap_to_primap2_unit_empty(caplog):
    assert pm2io.convert_unit_primap_to_primap2("", "FGASES") == "error__FGASES"
    assert "WARNING" in caplog.text
    assert "Input unit is empty. Nothing converted." in caplog.text


def test_convert_unit_primap_to_primap2_entity_empty(caplog):
    assert pm2io.convert_unit_primap_to_primap2("GgCO2eq", "") == "error_GgCO2eq_"
    assert "WARNING" in caplog.text
    assert "Input entity is empty. Nothing converted." in caplog.text


@pytest.mark.parametrize(
    "unit, entity, expected_attrs",
    [
        ("Mt", "CO2", {"units": "Mt", "entity": "CO2"}),
        (
            "Gg CO2",
            "KYOTOGHG (AR4GWP100)",
            {
                "units": "Gg CO2",
                "entity": "KYOTOGHG",
                "gwp_context": "AR4GWP100",
            },
        ),
        (
            "kg CO2",
            "CH4 (SARGWP100)",
            {
                "units": "kg CO2",
                "entity": "CH4",
                "gwp_context": "SARGWP100",
            },
        ),
    ],
)
def test_metadata_for_variable(unit, entity, expected_attrs):
    assert pm2io.metadata_for_variable(unit, entity) == expected_attrs


@pytest.mark.parametrize(
    "entity_pm1, entity_pm2",
    [
        ("CO2", "CO2"),
        ("KYOTOGHG", "KYOTOGHG (SARGWP100)"),
        ("KYOTOGHGAR4", "KYOTOGHG (AR4GWP100)"),
    ],
)
def test_convert_entity_gwp_primap(entity_pm1, entity_pm2):
    assert pm2io.convert_entity_gwp_primap(entity_pm1) == entity_pm2


def test_read_wide_csv_file():
    path = os.path.join("primap2", "tests", "data")
    file_input = "test_csv_data_sec_cat.csv"
    file_expected = "test_read_wide_csv_file_output.csv"
    df_expected = pd.read_csv(os.path.join(path, file_expected), index_col=0)

    coords_cols = {
        "unit": "unit",
        "entity": "gas",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "class",
    }
    coords_defaults = {
        "source": "TESTcsv2021",
        "citation": "Test",
        "sec_cats__Type": "fugitive",
        "scenario": "HISTORY",
    }
    coords_terminologies = {
        "area": "ISO3",
        "category": "IPCC2006",
        "sec_cats__Type": "type",
        "sec_cats__Class": "class",
        "scenario": "general",
    }
    meta_mapping = {"category": "PRIMAP1", "entity": "PRIMAP1"}
    filter_keep = {
        "f1": {"category": ["IPC0", "IPC2"]},
    }
    filter_remove = {
        "f1": {"gas": "CH4"},
    }

    df_result = pm2io.read_wide_csv_file_if(
        file_input,
        path,
        coords_cols=coords_cols,
        coords_defaults=coords_defaults,
        coords_terminologies=coords_terminologies,
        meta_mapping=meta_mapping,
        filter_keep=filter_keep,
        filter_remove=filter_remove,
    )
    df_result.to_csv(os.path.join(path, "temp.csv"))
    df_result = pd.read_csv(os.path.join(path, "temp.csv"), index_col=0)
    os.remove(os.path.join(path, "temp.csv"))
    pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)


def test_from_interchange_format():
    path = os.path.join("primap2", "tests", "data")
    file_input = "test_read_wide_csv_file_output.csv"
    file_expected = "test_from_interchange_format_output.nc"
    ds_expected = pm2.open_dataset(os.path.join(path, file_expected))
    attrs = {
        "area": "area (ISO3)",
        "cat": "category (IPCC2006)",
        "scen": "scenario (general)",
        "sec_cats": ["Class (class)", "Type (type)"],
    }
    df_input = pd.read_csv(os.path.join(path, file_input), index_col=0)
    ds_result = pm2io.from_interchange_format(
        df_input, attrs, data_col_regex_str="\d"  # noqa: W605
    )
    assert_ds_aligned_equal(ds_result, ds_expected, equal_nan=True)


def test_convert_dataframe_units_primap_to_primap2():
    path = os.path.join("primap2", "tests", "data")
    file_input = "test_csv_data_sec_cat.csv"
    file_expected = "test_convert_dataframe_units_primap_to_primap2.csv"
    df_expected = pd.read_csv(os.path.join(path, file_expected), index_col=0)
    df = pd.read_csv(os.path.join(path, file_input))
    df_converted = pm2io.convert_dataframe_units_primap_to_primap2(
        df, unit_col="unit", entity_col="gas"
    )
    pd.testing.assert_frame_equal(df_converted, df_expected)


# functions that still need individual testing

# dates_to_dimension(ds: xr.Dataset, time_format: str = "%Y") -> xr.DataArray:

# harmonize_units
