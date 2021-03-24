"""Tests for _data_reading.py"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import primap2 as pm2
import primap2.pm2io._data_reading as pm2io

from .utils import assert_ds_aligned_equal

DATA_PATH = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "code_in, expected_code_out",
    [
        ("IPC1A", "1.A"),
        ("CATM0EL", "M.0.EL"),
        ("IPC1A1B23", "1.A.1.b.ii.3"),
        ("IPCM1B1C", "M.1.B.1.c"),
    ],
)
def test_convert_ipcc_code_primap_to_primap2(code_in, expected_code_out):
    assert pm2io.convert_ipcc_code_primap_to_primap2(code_in) == expected_code_out


def test_convert_ipcc_code_primap_to_primap2_too_short(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC") == "error_IPC"
    assert "WARNING" in caplog.text
    assert "Too short to be a PRIMAP IPCC code." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_wrong_format(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPD1A") == "error_IPD1A"
    assert "WARNING" in caplog.text
    assert "Prefix is missing, must be one of 'IPC' or 'CAT'." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_first_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPCA1") == "error_IPCA1"
    assert "WARNING" in caplog.text
    assert "No digit found on first level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_second_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC123") == "error_IPC123"
    assert "WARNING" in caplog.text
    assert "No letter found on second level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_third_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC1AC") == "error_IPC1AC"
    assert "WARNING" in caplog.text
    assert "No number found on third level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_fourth_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC1A2_") == "error_IPC1A2_"
    assert "WARNING" in caplog.text
    assert "No letter found on fourth level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_fifth_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC1A2BB") == "error_IPC1A2BB"
    assert "WARNING" in caplog.text
    assert "No digit found on fifth level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_sixth_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC1A2B3X") == "error_IPC1A2B3X"
    assert "WARNING" in caplog.text
    assert "No number found on sixth level." in caplog.text


def test_convert_ipcc_code_primap_to_primap2_after_sixth_lvl(caplog):
    assert pm2io.convert_ipcc_code_primap_to_primap2("IPC1A2B33A") == "error_IPC1A2B33A"
    assert "WARNING" in caplog.text
    assert "Chars left after sixth level." in caplog.text


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


def test_read_wide_csv_file(tmp_path):
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"
    file_expected = DATA_PATH / "test_read_wide_csv_file_output.csv"
    df_expected = pd.read_csv(file_expected, index_col=0)

    coords_cols = {
        "unit": "unit",
        "entity": "gas",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "classification",
    }
    coords_defaults = {
        "source": "TESTcsv2021",
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
    coords_value_mapping = {"category": "PRIMAP1", "entity": "PRIMAP1"}
    filter_keep = {
        "f1": {"category": ["IPC0", "IPC2"]},
        "f2": {"classification": "TOTAL"},
    }
    filter_remove = {"f1": {"gas": "CH4"}, "f2": {"country": ["USA", "FRA"]}}
    meta_data = {"references": "Just ask around."}

    df_result = pm2io.read_wide_csv_file_if(
        file_input,
        coords_cols=coords_cols,
        coords_defaults=coords_defaults,
        coords_terminologies=coords_terminologies,
        coords_value_mapping=coords_value_mapping,
        filter_keep=filter_keep,
        filter_remove=filter_remove,
        meta_data=meta_data,
    )
    attrs_result = df_result.attrs
    df_result.to_csv(tmp_path / "temp.csv")
    df_result = pd.read_csv(tmp_path / "temp.csv", index_col=0)
    pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)

    assert attrs_result == {
        "references": "Just ask around.",
        "sec_cats": ["Class (class)", "Type (type)"],
        "scen": "scenario (general)",
        "area": "area (ISO3)",
        "cat": "category (IPCC2006)",
    }


def test_read_wide_csv_file_coords_value_mapping(tmp_path):
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"
    file_expected = DATA_PATH / "test_read_wide_csv_file_output.csv"
    df_expected = pd.read_csv(file_expected, index_col=0)

    coords_cols = {
        "unit": "unit",
        "entity": "gas",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "classification",
    }
    coords_defaults = {
        "source": "TESTcsv2021",
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
    coords_value_mapping = {
        "category": {"IPC1": "1", "IPC2": "2", "IPC3": "3", "IPC0": "0"},
        "entity": {"KYOTOGHG": "KYOTOGHG (SARGWP100)"},
        "unit": "PRIMAP1",
    }
    filter_keep = {
        "f1": {"category": ["IPC0", "IPC2"]},
        "f2": {"classification": "TOTAL"},
    }
    filter_remove = {"f1": {"gas": "CH4"}, "f2": {"country": ["USA", "FRA"]}}

    df_result = pm2io.read_wide_csv_file_if(
        file_input,
        coords_cols=coords_cols,
        coords_defaults=coords_defaults,
        coords_terminologies=coords_terminologies,
        coords_value_mapping=coords_value_mapping,
        filter_keep=filter_keep,
        filter_remove=filter_remove,
    )
    df_result.to_csv(tmp_path / "temp.csv")
    df_result = pd.read_csv(tmp_path / "temp.csv", index_col=0)
    pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)


def test_read_wide_csv_file_entity_def(tmp_path):
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"
    file_expected = DATA_PATH / "test_read_wide_csv_file_output_entity_def.csv"
    df_expected = pd.read_csv(file_expected, index_col=0)

    coords_cols = {
        "unit": "unit",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "classification",
    }
    coords_defaults = {
        "source": "TESTcsv2021",
        "sec_cats__Type": "fugitive",
        "scenario": "HISTORY",
        "entity": "CO2",
    }
    coords_terminologies = {
        "area": "ISO3",
        "category": "IPCC2006",
        "sec_cats__Type": "type",
        "sec_cats__Class": "class",
        "scenario": "general",
    }
    meta_mapping = {
        "category": "PRIMAP1",
    }
    filter_keep = {
        "f1": {"category": ["IPC0", "IPC2"]},
        "f2": {"classification": "TOTAL"},
    }
    filter_remove = {"f2": {"country": ["USA", "FRA"]}}

    df_result = pm2io.read_wide_csv_file_if(
        file_input,
        coords_cols=coords_cols,
        coords_defaults=coords_defaults,
        coords_terminologies=coords_terminologies,
        coords_value_mapping=meta_mapping,
        filter_keep=filter_keep,
        filter_remove=filter_remove,
    )
    df_result.to_csv(tmp_path / "temp.csv")
    df_result = pd.read_csv(tmp_path / "temp.csv", index_col=0)
    pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)


def test_read_wide_csv_file_unit_def(tmp_path):
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"
    file_expected = DATA_PATH / "test_read_wide_csv_file_output_unit_def.csv"
    df_expected = pd.read_csv(file_expected, index_col=0)

    coords_cols = {
        "entity": "gas",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "classification",
    }
    coords_defaults = {
        "source": "TESTcsv2021",
        "sec_cats__Type": "fugitive",
        "scenario": "HISTORY",
        "unit": "Gg",
    }
    coords_terminologies = {
        "area": "ISO3",
        "category": "IPCC2006",
        "sec_cats__Type": "type",
        "sec_cats__Class": "class",
        "scenario": "general",
    }
    meta_mapping = {
        "category": "PRIMAP1",
    }
    filter_keep = {
        "f1": {"category": ["IPC0", "IPC2"]},
        "f2": {"classification": "TOTAL"},
    }
    filter_remove = {"f2": {"country": ["USA", "FRA"]}, "f1": {"gas": "KYOTOGHG"}}

    df_result = pm2io.read_wide_csv_file_if(
        file_input,
        coords_cols=coords_cols,
        coords_defaults=coords_defaults,
        coords_terminologies=coords_terminologies,
        coords_value_mapping=meta_mapping,
        filter_keep=filter_keep,
        filter_remove=filter_remove,
    )
    df_result.to_csv(tmp_path / "test.csv")
    df_result = pd.read_csv(tmp_path / "test.csv", index_col=0)
    pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)


def test_read_wide_csv_file_col_missing():
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

    coords_cols = {
        "unit": "unit",
        "entity": "gas",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "class",
    }
    coords_defaults = {
        "source": "TESTcsv2021",
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
    coords_value_mapping = {"category": "PRIMAP1", "entity": "PRIMAP1"}
    filter_keep = {
        "f1": {"category": ["IPC0", "IPC2"]},
        "f2": {"classification": "TOTAL"},
    }
    filter_remove = {"f1": {"gas": "CH4"}, "f2": {"country": ["USA", "FRA"]}}

    with pytest.raises(ValueError, match="Columns {'class'} not found in CSV."):
        pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
            filter_keep=filter_keep,
            filter_remove=filter_remove,
        )


def test_read_wide_csv_file_no_unit():
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

    coords_cols = {
        "entity": "gas",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "classification",
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

    with pytest.raises(ValueError):
        pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=meta_mapping,
        )


def test_read_wide_csv_file_mapping_not_implemented_for_col():
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

    coords_cols = {
        "unit": "unit",
        "entity": "gas",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "classification",
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
    meta_mapping = {"area": "PRIMAP1", "entity": "PRIMAP1"}

    with pytest.raises(ValueError):
        pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=meta_mapping,
        )


def test_read_wide_csv_file_mandatory_missing():
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

    coords_cols = {
        "unit": "unit",
        "entity": "gas",
        "category": "category",
        "sec_cats__Class": "classification",
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

    with pytest.raises(ValueError):
        pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=meta_mapping,
        )


def test_read_wide_csv_file_no_entity():
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

    coords_cols = {
        "unit": "unit",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "classification",
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

    with pytest.raises(ValueError):
        pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=meta_mapping,
        )


def test_read_wide_csv_file_unknown_cat_mapping():
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

    coords_cols = {
        "unit": "unit",
        "entity": "gas",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "classification",
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
    meta_mapping = {"category": "TESTTEST", "entity": "PRIMAP1"}

    with pytest.raises(ValueError):
        pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=meta_mapping,
        )


def test_read_wide_csv_file_unknown_entity_mapping():
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

    coords_cols = {
        "unit": "unit",
        "entity": "gas",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "classification",
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
    meta_mapping = {"category": "PRIMAP1", "entity": "TESTTEST"}

    with pytest.raises(ValueError):
        pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=meta_mapping,
        )


def test_read_wide_csv_file_no_function_mapping_col():
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

    coords_cols = {
        "entity": "gas",
        "area": "country",
        "sec_cats__Class": "classification",
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
    meta_mapping = {"category": "PRIMAP1", "entity": "PRIMAP1", "area": "TESTTEST"}

    with pytest.raises(ValueError):
        pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=meta_mapping,
        )


def test_from_interchange_format():
    file_input = DATA_PATH / "test_read_wide_csv_file_output.csv"
    file_expected = DATA_PATH / "test_from_interchange_format_output.nc"
    ds_expected = pm2.open_dataset(file_expected)
    attrs = {
        "area": "area (ISO3)",
        "cat": "category (IPCC2006)",
        "scen": "scenario (general)",
        "sec_cats": ["Class (class)", "Type (type)"],
    }
    df_input = pd.read_csv(file_input, index_col=0)
    ds_result = pm2io.from_interchange_format(df_input, attrs, time_col_regex=r"\d")
    assert_ds_aligned_equal(ds_result, ds_expected, equal_nan=True)


def test_from_interchange_format_too_large(caplog):
    df = pd.DataFrame(
        {
            "a": np.arange(10),
            "b": np.arange(10),
            "c": np.arange(10),
            "unit": ["Gg"] * 10,
            "2001": np.arange(10),
        }
    )
    df.attrs = {}
    # projected array size should be 1000 > 100
    with pytest.raises(ValueError, match="Resulting array too large"):
        pm2io.from_interchange_format(df, max_array_size=100)

    assert "ERROR" in caplog.text
    assert (
        "Array with 2 dimensions will have a size of 1,000 due to the shape"
        " [10, 10, 10]." in caplog.text
    )


def test_convert_dataframe_units_primap_to_primap2():
    file_input = DATA_PATH / "test_csv_data_sec_cat.csv"
    file_expected = DATA_PATH / "test_convert_dataframe_units_primap_to_primap2.csv"
    df_expected = pd.read_csv(file_expected, index_col=0)
    df = pd.read_csv(file_input)
    df_converted = pm2io.convert_dataframe_units_primap_to_primap2(
        df, unit_col="unit", entity_col="gas"
    )
    pd.testing.assert_frame_equal(df_converted, df_expected)


def test_read_write_interchange_format_roundtrip():
    file_input = DATA_PATH / "test_read_wide_csv_file_output.csv"
    file_temp = DATA_PATH / "test_interchange_format"
    data = pd.read_csv(file_input, index_col=0)
    attrs = {
        "area": "area (ISO3)",
        "cat": "category (IPCC2006)",
        "scen": "scenario (general)",
        "sec_cats": ["Class (class)", "Type (type)"],
    }
    pm2.pm2io.write_interchange_format(file_temp, data, attrs)
    read_data = pm2.pm2io.read_interchange_format(file_temp)
    yaml_path = file_temp.parent / (file_temp.stem + ".yaml")
    yaml_path.unlink()
    csv_path = file_temp.parent / (file_temp.stem + ".csv")
    csv_path.unlink()
    read_attrs = read_data.attrs
    assert read_attrs == attrs
    pd.testing.assert_frame_equal(data, read_data)


# functions that still need individual testing

# dates_to_dimension(ds: xr.Dataset, time_format: str = "%Y") -> xr.DataArray:

# harmonize_units
