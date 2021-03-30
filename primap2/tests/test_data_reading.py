"""Tests for _data_reading.py"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import primap2
import primap2.pm2io as pm2io

from .utils import assert_ds_aligned_equal

DATA_PATH = Path(__file__).parent / "data"


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
    assert (
        pm2io._interchange_format.metadata_for_variable(unit, entity) == expected_attrs
    )


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
    coords_value_mapping = {
        "category": "PRIMAP1",
        "entity": "PRIMAP1",
        "unit": "PRIMAP1",
    }
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

    attrs_expected = {
        "attrs": {
            "references": "Just ask around.",
            "sec_cats": ["Class (class)", "Type (type)"],
            "scen": "scenario (general)",
            "area": "area (ISO3)",
            "cat": "category (IPCC2006)",
        },
        "time_format": "%Y",
        "dimensions": {
            ent: [
                "entity",
                "source",
                "area (ISO3)",
                "Type (type)",
                "unit",
                "scenario (general)",
                "Class (class)",
                "category (IPCC2006)",
            ]
            for ent in np.unique(df_expected["entity"])
        },
    }

    assert attrs_result.keys() == attrs_expected.keys()
    assert attrs_result["attrs"] == attrs_expected["attrs"]
    assert attrs_result["time_format"] == attrs_expected["time_format"]
    assert attrs_result["dimensions"].keys() == attrs_expected["dimensions"].keys()
    for entity in attrs_result["dimensions"]:
        assert set(attrs_result["dimensions"][entity]) == set(
            attrs_expected["dimensions"][entity]
        )


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
        "unit": "PRIMAP1",
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
        "unit": "PRIMAP1",
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
    ds_expected = primap2.open_dataset(file_expected)
    df_input = pd.read_csv(file_input, index_col=0)
    dims = [
        "area (ISO3)",
        "category (IPCC2006)",
        "scenario (general)",
        "Class (class)",
        "Type (type)",
        "unit",
        "entity",
        "source",
    ]
    attrs = {
        "attrs": {
            "area": "area (ISO3)",
            "cat": "category (IPCC2006)",
            "scen": "scenario (general)",
            "sec_cats": ["Class (class)", "Type (type)"],
        },
        "time_format": "%Y",
        "dimensions": {ent: dims for ent in np.unique(df_input["entity"])},
    }
    ds_result = pm2io.from_interchange_format(df_input, attrs)
    assert_ds_aligned_equal(ds_result, ds_expected, equal_nan=True)


def test_from_interchange_format_too_large(caplog):
    df = pd.DataFrame(
        {
            "a": np.arange(10),
            "b": np.arange(10),
            "c": np.arange(10),
            "entity": ["CO2"] * 10,
            "unit": ["Gg"] * 10,
            "2001": np.arange(10),
        }
    )
    df.attrs = {
        "attrs": {},
        "dimensions": {"CO2": ["a", "b", "c"]},
        "time_format": "%Y",
    }
    # projected array size should be 1000 > 100
    with pytest.raises(ValueError, match="Resulting array too large"):
        pm2io.from_interchange_format(df, max_array_size=100)

    assert "ERROR" in caplog.text
    assert (
        "Set with 1 entities and a total of 3 dimensions will have a size of 1,000"
        in caplog.text
    )


def test_read_write_interchange_format_roundtrip(tmp_path):
    file_input = DATA_PATH / "test_read_wide_csv_file_output.csv"
    file_temp = tmp_path / "test_interchange_format"
    data = pd.read_csv(file_input, index_col=0)
    attrs = {
        "attrs": {
            "area": "area (ISO3)",
            "cat": "category (IPCC2006)",
            "scen": "scenario (general)",
            "sec_cats": ["Class (class)", "Type (type)"],
        },
        "time_format": "%Y",
        "dimensions": {"CO2": ["area (ISO3)"]},
    }
    pm2io.write_interchange_format(file_temp, data, attrs)
    read_data = pm2io.read_interchange_format(file_temp)
    read_attrs = read_data.attrs
    assert read_attrs == attrs
    pd.testing.assert_frame_equal(data, read_data)


# functions that still need individual testing

# dates_to_dimension(ds: xr.Dataset, time_format: str = "%Y") -> xr.DataArray:

# harmonize_units
