"""Tests for _data_reading.py"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import primap2
import primap2.pm2io as pm2io
import primap2.pm2io._conversion
from primap2.pm2io._data_reading import additional_coordinate_metadata

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


@pytest.mark.parametrize(
    "to_test_for_float, expected_result",
    [
        ("0.15", True),
        ("IE", False),
        ("NaN", True),
        (
            "25,000",
            False,
        ),  # be careful when reading data to process thousands seperators
        ("25,00", False),
        ("IE, NO", False),
    ],
)
def test_is_float(to_test_for_float, expected_result):
    assert pm2io._data_reading.is_float(to_test_for_float) == expected_result


@pytest.mark.parametrize(
    "code_to_test, expected_result",
    [
        ("IE", 0),
        ("NO", 0),
        ("-", 0),
        ("NO,NE", 0),
        ("NO, NE", 0),
        ("NE,NO", 0),
        ("NE, NO", 0),
        ("IE, NE, NO", 0),
        ("IE,NA,NO", 0),
        ("NA,IE,NO", 0),
        ("NO,NE,IE", 0),
    ],
)
def test_parse_code(code_to_test, expected_result):
    assert pm2io._data_reading.parse_code(code_to_test) == expected_result


@pytest.mark.parametrize(
    "code_to_test_nan, expected_result",
    [
        ("NE", np.nan),
        ("NE0", np.nan),
        ("C", np.nan),
        ("NaN", np.nan),
        ("nan", np.nan),
        ("NA, NE", np.nan),
        ("NA,NE", np.nan),
    ],
)
def test_parse_code_nan(code_to_test_nan, expected_result):
    assert np.isnan(pm2io._data_reading.parse_code(code_to_test_nan))


@pytest.mark.parametrize(
    "strs, user_na_conv, expected_result",
    [
        (
            ["IE", "IE,NA", "NA"],
            {},
            {"IE": 0, "IE,NA": 0, "NA": np.nan},
        ),
        (
            ["IE", "IE,NA", "NA"],
            {"NA": 0},
            {"IE": 0, "IE,NA": 0, "NA": 0},
        ),
    ],
)
def test_create_na_replacement_dict(strs, user_na_conv, expected_result):
    assert (
        pm2io._data_reading.create_str_replacement_dict(strs, user_na_conv)
        == expected_result
    )


def assert_attrs_equal(attrs_result, attrs_expected):
    assert attrs_result.keys() == attrs_expected.keys()
    assert attrs_result["attrs"] == attrs_expected["attrs"]
    assert attrs_result["time_format"] == attrs_expected["time_format"]
    assert attrs_result["dimensions"].keys() == attrs_expected["dimensions"].keys()
    for entity in attrs_result["dimensions"]:
        assert set(attrs_result["dimensions"][entity]) == set(
            attrs_expected["dimensions"][entity]
        )


@pytest.fixture
def coords_cols():
    return {
        "unit": "unit",
        "entity": "gas",
        "area": "country",
        "category": "category",
        "sec_cats__Class": "classification",
    }


@pytest.fixture
def add_coords_cols():
    return {"category_name": ["category_name", "category"]}


@pytest.fixture
def coords_defaults():
    return {
        "source": "TESTcsv2021",
        "sec_cats__Type": "fugitive",
        "scenario": "HISTORY",
    }


@pytest.fixture
def coords_terminologies():
    return {
        "area": "ISO3",
        "category": "IPCC2006",
        "sec_cats__Type": "type",
        "sec_cats__Class": "class",
        "scenario": "general",
    }


@pytest.fixture
def coords_value_mapping():
    return {
        "category": "PRIMAP1",
        "entity": "PRIMAP1",
        "unit": "PRIMAP1",
    }


@pytest.fixture
def coords_value_filling():
    return {
        "category": {  # col to fill
            "category_name": {  # col to fill from
                "Energy": "1",  # from value: to value
                "IPPU": "2",
            }
        }
    }


@pytest.fixture
def filter_keep():
    return {
        "f1": {"category": ["IPC0", "IPC2"]},
        "f2": {"classification": "TOTAL"},
    }


@pytest.fixture
def filter_remove():
    return {"f1": {"gas": "CH4"}, "f2": {"country": ["USA", "FRA"]}}


class TestReadWideCSVFile:
    def test_output(
        self,
        tmp_path,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
        filter_keep,
        filter_remove,
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"
        file_expected = DATA_PATH / "test_read_wide_csv_file_output.csv"
        df_expected = pd.read_csv(file_expected, index_col=0)

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
                "*": [
                    "entity",
                    "source",
                    "area (ISO3)",
                    "Type (type)",
                    "unit",
                    "scenario (general)",
                    "Class (class)",
                    "category (IPCC2006)",
                ]
            },
        }

        assert_attrs_equal(attrs_result, attrs_expected)

    def test_no_sec_cats(
        self,
        tmp_path,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
    ):
        file_input = DATA_PATH / "test_csv_data.csv"
        file_expected = DATA_PATH / "test_read_wide_csv_file_no_sec_cats.csv"
        df_expected = pd.read_csv(file_expected, index_col=0)

        del coords_cols["sec_cats__Class"]
        del coords_defaults["sec_cats__Type"]
        del coords_terminologies["sec_cats__Class"]
        del coords_terminologies["sec_cats__Type"]

        df_result = pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
        )
        attrs_result = df_result.attrs
        df_result.to_csv(tmp_path / "temp.csv")
        df_result = pd.read_csv(tmp_path / "temp.csv", index_col=0)
        pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)

        attrs_expected = {
            "attrs": {
                "scen": "scenario (general)",
                "area": "area (ISO3)",
                "cat": "category (IPCC2006)",
            },
            "time_format": "%Y",
            "dimensions": {
                "*": [
                    "entity",
                    "source",
                    "area (ISO3)",
                    "unit",
                    "scenario (general)",
                    "category (IPCC2006)",
                ]
            },
        }

        assert_attrs_equal(attrs_result, attrs_expected)

    def test_add_coords(
        self,
        tmp_path,
        coords_cols,
        add_coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
    ):
        file_input = DATA_PATH / "test_csv_data_category_name.csv"
        file_expected = DATA_PATH / "test_read_wide_csv_file_no_sec_cats_cat_name.csv"
        df_expected = pd.read_csv(file_expected, index_col=0)

        del coords_cols["sec_cats__Class"]
        del coords_defaults["sec_cats__Type"]
        del coords_terminologies["sec_cats__Class"]
        del coords_terminologies["sec_cats__Type"]

        df_result = pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            add_coords_cols=add_coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
        )
        attrs_result = df_result.attrs
        df_result.to_csv(tmp_path / "temp.csv")
        df_result = pd.read_csv(tmp_path / "temp.csv", index_col=0)
        pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)

        attrs_expected = {
            "attrs": {
                "scen": "scenario (general)",
                "area": "area (ISO3)",
                "cat": "category (IPCC2006)",
            },
            "time_format": "%Y",
            "dimensions": {
                "*": [
                    "entity",
                    "source",
                    "area (ISO3)",
                    "unit",
                    "scenario (general)",
                    "category (IPCC2006)",
                ]
            },
            "additional_coordinates": {"category_name": "category (IPCC2006)"},
        }

        assert_attrs_equal(attrs_result, attrs_expected)

    def test_read_wide_fill_col(
        self,
        tmp_path,
        coords_cols,
        add_coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
        coords_value_filling,
    ):
        file_input = DATA_PATH / "test_csv_data_category_name_fill_cat_code.csv"
        file_expected = DATA_PATH / "test_read_wide_csv_file_no_sec_cats_cat_name.csv"
        df_expected = pd.read_csv(file_expected, index_col=0)

        del coords_cols["sec_cats__Class"]
        del coords_defaults["sec_cats__Type"]
        del coords_terminologies["sec_cats__Class"]
        del coords_terminologies["sec_cats__Type"]

        df_result = pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            add_coords_cols=add_coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
            coords_value_filling=coords_value_filling,
        )
        attrs_result = df_result.attrs
        df_result.to_csv(tmp_path / "temp.csv")
        df_result = pd.read_csv(tmp_path / "temp.csv", index_col=0)
        pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)

        attrs_expected = {
            "attrs": {
                "scen": "scenario (general)",
                "area": "area (ISO3)",
                "cat": "category (IPCC2006)",
            },
            "time_format": "%Y",
            "dimensions": {
                "*": [
                    "entity",
                    "source",
                    "area (ISO3)",
                    "unit",
                    "scenario (general)",
                    "category (IPCC2006)",
                ]
            },
            "additional_coordinates": {"category_name": "category (IPCC2006)"},
        }

        assert_attrs_equal(attrs_result, attrs_expected)

    def test_entity_terminology(
        self,
        tmp_path,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
    ):
        file_input = DATA_PATH / "test_csv_data.csv"
        file_expected = DATA_PATH / "test_read_wide_csv_file_no_sec_cats.csv"
        df_expected: pd.DataFrame = pd.read_csv(file_expected, index_col=0)
        df_expected = df_expected.rename(columns={"entity": "entity (PRIMAP1)"})

        del coords_cols["sec_cats__Class"]
        del coords_defaults["sec_cats__Type"]
        del coords_terminologies["sec_cats__Class"]
        del coords_terminologies["sec_cats__Type"]

        coords_terminologies["entity"] = "PRIMAP1"

        df_result = pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
        )
        attrs_result = df_result.attrs
        df_result.to_csv(tmp_path / "temp.csv")
        df_result = pd.read_csv(tmp_path / "temp.csv", index_col=0)
        pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)

        attrs_expected = {
            "attrs": {
                "scen": "scenario (general)",
                "area": "area (ISO3)",
                "cat": "category (IPCC2006)",
                "entity_terminology": "PRIMAP1",
            },
            "time_format": "%Y",
            "dimensions": {
                "*": [
                    "entity (PRIMAP1)",
                    "source",
                    "area (ISO3)",
                    "unit",
                    "scenario (general)",
                    "category (IPCC2006)",
                ]
            },
        }

        assert_attrs_equal(attrs_result, attrs_expected)

    def test_coords_value_mapping_dict(
        self,
        tmp_path,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        filter_keep,
        filter_remove,
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"
        file_expected = DATA_PATH / "test_read_wide_csv_file_output.csv"
        df_expected = pd.read_csv(file_expected, index_col=0)

        coords_value_mapping = {
            "category": {"IPC1": "1", "IPC2": "2", "IPC3": "3", "IPC0": "0"},
            "entity": {"KYOTOGHG": "KYOTOGHG (SARGWP100)"},
            "unit": "PRIMAP1",
        }

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

    def test_entity_default(
        self,
        tmp_path,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
        filter_keep,
        filter_remove,
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"
        file_expected = DATA_PATH / "test_read_wide_csv_file_output_entity_def.csv"
        df_expected = pd.read_csv(file_expected, index_col=0)

        del coords_cols["entity"]
        del coords_value_mapping["entity"]
        coords_defaults["entity"] = "CO2"
        del filter_remove["f1"]

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

    def test_unit_default(
        self,
        tmp_path,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
        filter_keep,
        filter_remove,
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"
        file_expected = DATA_PATH / "test_read_wide_csv_file_output_unit_def.csv"
        df_expected = pd.read_csv(file_expected, index_col=0)

        del coords_cols["unit"]
        coords_defaults["unit"] = "Gg"
        filter_remove["f1"] = {"gas": "KYOTOGHG"}

        df_result = pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
            filter_keep=filter_keep,
            filter_remove=filter_remove,
        )
        df_result.to_csv(tmp_path / "test.csv")
        df_result = pd.read_csv(tmp_path / "test.csv", index_col=0)
        pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)

    def test_unit_harmonization(
        self,
        tmp_path,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
    ):
        file_input = DATA_PATH / "test_csv_data_unit_harmonization.csv"
        file_expected = DATA_PATH / "test_read_wide_csv_file_output_unit_harm.csv"
        df_expected = pd.read_csv(file_expected, index_col=0)

        del coords_cols["sec_cats__Class"]
        del coords_defaults["sec_cats__Type"]
        del coords_terminologies["sec_cats__Class"]
        del coords_terminologies["sec_cats__Type"]

        df_result = pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
        )
        df_result.to_csv(tmp_path / "test.csv")
        df_result = pd.read_csv(tmp_path / "test.csv", index_col=0)
        pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)

    def test_function_mapping(
        self,
        tmp_path,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
        filter_keep,
        filter_remove,
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"
        file_expected = DATA_PATH / "test_read_wide_csv_file_output_unit_def.csv"
        df_expected = pd.read_csv(file_expected, index_col=0)

        del coords_cols["unit"]
        coords_defaults["unit"] = "Gg"
        coords_value_mapping["category"] = (
            pm2io._conversion.convert_ipcc_code_primap_to_primap2
        )
        filter_remove["f1"] = {"gas": "KYOTOGHG"}

        df_result = pm2io.read_wide_csv_file_if(
            file_input,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
            filter_keep=filter_keep,
            filter_remove=filter_remove,
        )
        df_result.to_csv(tmp_path / "test.csv")
        df_result = pd.read_csv(tmp_path / "test.csv", index_col=0)
        pd.testing.assert_frame_equal(df_result, df_expected, check_column_type=False)

    def test_col_missing(
        self,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
        filter_keep,
        filter_remove,
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

        coords_cols["sec_cats__Class"] = "class"

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

    def test_unknown_mapping(
        self, coords_cols, coords_defaults, coords_terminologies, coords_value_mapping
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

        coords_value_mapping["category"] = "non-existing"

        with pytest.raises(
            ValueError,
            match="Unknown metadata mapping 'non-existing' for column 'category'.",
        ):
            pm2io.read_wide_csv_file_if(
                file_input,
                coords_cols=coords_cols,
                coords_defaults=coords_defaults,
                coords_terminologies=coords_terminologies,
                coords_value_mapping=coords_value_mapping,
            )

    def test_overlapping_specification(
        self, coords_cols, coords_defaults, coords_terminologies, coords_value_mapping
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

        coords_defaults["entity"] = "CO2"

        with pytest.raises(
            ValueError, match="{'entity'} given in coords_cols and coords_defaults."
        ):
            pm2io.read_wide_csv_file_if(
                file_input,
                coords_cols=coords_cols,
                coords_defaults=coords_defaults,
                coords_terminologies=coords_terminologies,
                coords_value_mapping=coords_value_mapping,
            )

    def test_overlapping_specification_add_coords(
        self, coords_cols, coords_defaults, coords_terminologies, coords_value_mapping
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

        add_coords_cols = {"test": ["gas", "category"]}

        with pytest.raises(
            ValueError, match="{'gas'} given in coords_cols and add_coords_cols."
        ):
            pm2io.read_wide_csv_file_if(
                file_input,
                coords_cols=coords_cols,
                add_coords_cols=add_coords_cols,
                coords_defaults=coords_defaults,
                coords_terminologies=coords_terminologies,
                coords_value_mapping=coords_value_mapping,
            )

    def test_no_unit(
        self, coords_cols, coords_defaults, coords_terminologies, coords_value_mapping
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

        del coords_cols["unit"]

        with pytest.raises(ValueError, match="Mandatory dimension 'unit' not defined."):
            pm2io.read_wide_csv_file_if(
                file_input,
                coords_cols=coords_cols,
                coords_defaults=coords_defaults,
                coords_terminologies=coords_terminologies,
                coords_value_mapping=coords_value_mapping,
            )

    def test_no_entity(
        self, coords_cols, coords_defaults, coords_terminologies, coords_value_mapping
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

        del coords_cols["entity"]

        with pytest.raises(
            ValueError, match="Mandatory dimension 'entity' not defined."
        ):
            pm2io.read_wide_csv_file_if(
                file_input,
                coords_cols=coords_cols,
                coords_defaults=coords_defaults,
                coords_terminologies=coords_terminologies,
                coords_value_mapping=coords_value_mapping,
            )

    def test_unknown_category_mapping(
        self, coords_cols, coords_defaults, coords_terminologies, coords_value_mapping
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

        coords_value_mapping["category"] = "TESTTEST"

        with pytest.raises(
            ValueError,
            match="Unknown metadata mapping 'TESTTEST' for column 'category'.",
        ):
            pm2io.read_wide_csv_file_if(
                file_input,
                coords_cols=coords_cols,
                coords_defaults=coords_defaults,
                coords_terminologies=coords_terminologies,
                coords_value_mapping=coords_value_mapping,
            )

    def test_unknown_entity_mapping(
        self, coords_cols, coords_defaults, coords_terminologies, coords_value_mapping
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

        coords_value_mapping["entity"] = "TESTTEST"

        with pytest.raises(
            ValueError, match="Unknown metadata mapping 'TESTTEST' for column 'entity'."
        ):
            pm2io.read_wide_csv_file_if(
                file_input,
                coords_cols=coords_cols,
                coords_defaults=coords_defaults,
                coords_terminologies=coords_terminologies,
                coords_value_mapping=coords_value_mapping,
            )

    def test_unknown_coordinate(
        self, coords_cols, coords_defaults, coords_terminologies, coords_value_mapping
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat.csv"

        coords_defaults["citation"] = "this should go to attrs"

        with pytest.raises(
            ValueError,
            match="'citation' given in coords_defaults is unknown - prefix with "
            "'sec_cats__' to add a secondary category.",
        ):
            pm2io.read_wide_csv_file_if(
                file_input,
                coords_cols=coords_cols,
                coords_defaults=coords_defaults,
                coords_terminologies=coords_terminologies,
                coords_value_mapping=coords_value_mapping,
            )

    def test_unprocessed_strs(
        self,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
    ):
        file_input = DATA_PATH / "test_csv_data_sec_cat_strings.csv"

        with pytest.raises(ValueError, match="String values"):
            pm2io.read_wide_csv_file_if(
                file_input,
                coords_cols=coords_cols,
                coords_defaults=coords_defaults,
                coords_terminologies=coords_terminologies,
                coords_value_mapping=coords_value_mapping,
                filter_keep={},
                filter_remove={},
                convert_str=False,
            )


class TestInterchangeFormat:
    def test_from(self):
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
            "dimensions": {"*": dims},
        }
        ds_result = pm2io.from_interchange_format(df_input, attrs)
        assert_ds_aligned_equal(ds_result, ds_expected, equal_nan=True)

    def test_from_too_large(self, caplog):
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

    def test_from_add_coord_non_unique(self, caplog):
        df = pd.DataFrame(
            {
                "a": np.arange(3),
                "b": np.arange(3),
                "c": np.arange(3),
                "entity": ["CO2"] * 3,
                "entity_name": ["Carbondioxide", "Carbondioxide", "Methane"],
                "unit": ["Gg"] * 3,
                "2001": np.arange(3),
            }
        )
        df.attrs = {
            "attrs": {},
            "dimensions": {"CO2": ["a", "b", "c"]},
            "time_format": "%Y",
            "additional_coordinates": {"entity_name": "entity"},
        }

        with pytest.raises(
            ValueError,
            match="Different secondary coordinate values "
            "for given first coordinate value for "
            "entity_name.",
        ):
            pm2io.from_interchange_format(df)

        assert "ERROR" in caplog.text
        assert (
            "Different secondary coordinate values for given first coordinate "
            "value for entity_name." in caplog.text
        )

    def test_roundtrip(self, tmp_path):
        file_input = DATA_PATH / "test_read_wide_csv_file_output.csv"
        file_temp = tmp_path / "test_interchange_format"
        data = pd.read_csv(file_input, index_col=0, dtype=object)
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


class TestLong:
    def test_compare_wide(
        self,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
    ):
        file_input_wide = DATA_PATH / "test_csv_data.csv"
        file_input_long = DATA_PATH / "long.csv"

        del coords_cols["sec_cats__Class"]
        del coords_defaults["sec_cats__Type"]
        del coords_terminologies["sec_cats__Class"]
        del coords_terminologies["sec_cats__Type"]

        meta_data = {"references": "Just ask around"}

        df_result_wide = pm2io.read_wide_csv_file_if(
            file_input_wide,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
            meta_data=meta_data,
        )

        coords_cols["time"] = "year"
        coords_cols["data"] = "emissions"
        df_result_long = pm2io.read_long_csv_file_if(
            file_input_long,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
            meta_data=meta_data,
            time_format="%Y",
        )

        pd.testing.assert_frame_equal(df_result_wide, df_result_long)
        assert df_result_wide.attrs == df_result_long.attrs

    def test_compare_wide_entity_terminology(
        self,
        coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
    ):
        file_input_wide = DATA_PATH / "test_csv_data.csv"
        file_input_long = DATA_PATH / "long.csv"

        del coords_cols["sec_cats__Class"]
        del coords_defaults["sec_cats__Type"]
        del coords_terminologies["sec_cats__Class"]
        del coords_terminologies["sec_cats__Type"]

        coords_terminologies["entity"] = "PRIMAP1"

        df_result_wide = pm2io.read_wide_csv_file_if(
            file_input_wide,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
        )

        coords_cols["time"] = "year"
        coords_cols["data"] = "emissions"
        df_result_long = pm2io.read_long_csv_file_if(
            file_input_long,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
            time_format="%Y",
        )

        pd.testing.assert_frame_equal(df_result_wide, df_result_long)
        assert df_result_wide.attrs == df_result_long.attrs

    def test_compare_wide_add_cols(
        self,
        coords_cols,
        add_coords_cols,
        coords_defaults,
        coords_terminologies,
        coords_value_mapping,
    ):
        file_input_wide = DATA_PATH / "test_csv_data_category_name.csv"
        file_input_long = DATA_PATH / "test_csv_data_category_name_long.csv"

        del coords_cols["sec_cats__Class"]
        del coords_defaults["sec_cats__Type"]
        del coords_terminologies["sec_cats__Class"]
        del coords_terminologies["sec_cats__Type"]

        df_result_wide = pm2io.read_wide_csv_file_if(
            file_input_wide,
            coords_cols=coords_cols,
            add_coords_cols=add_coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
        )

        coords_cols["time"] = "year"
        coords_cols["data"] = "emissions"
        df_result_long = pm2io.read_long_csv_file_if(
            file_input_long,
            coords_cols=coords_cols,
            add_coords_cols=add_coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
            time_format="%Y",
        )

        pd.testing.assert_frame_equal(df_result_wide, df_result_long)
        assert df_result_wide.attrs == df_result_long.attrs

    def test_no_data_specified(
        self, coords_cols, coords_defaults, coords_terminologies, coords_value_mapping
    ):
        file_input_long = DATA_PATH / "long.csv"

        del coords_cols["sec_cats__Class"]
        del coords_defaults["sec_cats__Type"]
        del coords_terminologies["sec_cats__Class"]
        del coords_terminologies["sec_cats__Type"]

        coords_cols["time"] = "year"

        with pytest.raises(
            ValueError,
            match="No data column in the CSV specified in coords_cols, so nothing to"
            " read.",
        ):
            pm2io.read_long_csv_file_if(
                file_input_long,
                coords_cols=coords_cols,
                coords_defaults=coords_defaults,
                coords_terminologies=coords_terminologies,
                coords_value_mapping=coords_value_mapping,
                time_format="%Y",
            )

    def test_no_time(
        self, coords_cols, coords_defaults, coords_terminologies, coords_value_mapping
    ):
        file_input_long = DATA_PATH / "long_no_time.csv"

        del coords_cols["sec_cats__Class"]
        del coords_defaults["sec_cats__Type"]
        del coords_terminologies["sec_cats__Class"]
        del coords_terminologies["sec_cats__Type"]

        coords_cols["data"] = "emissions"
        coords_defaults["time"] = datetime.datetime(2020, 1, 1)

        df = pm2io.read_long_csv_file_if(
            file_input_long,
            coords_cols=coords_cols,
            coords_defaults=coords_defaults,
            coords_terminologies=coords_terminologies,
            coords_value_mapping=coords_value_mapping,
            time_format="%Y",
        )

        df_expected = pd.DataFrame(
            [
                ("TESTcsv2021", "HISTORY", "AUS", "CO2", "Gg CO2 / yr", "1", 4.1),
                ("TESTcsv2021", "HISTORY", "ZAM", "CH4", "Mt CH4 / yr", "2", 7.0),
            ],
            columns=[
                "source",
                "scenario (general)",
                "area (ISO3)",
                "entity",
                "unit",
                "category (IPCC2006)",
                "2020",
            ],
        )

        pd.testing.assert_frame_equal(df, df_expected)


class TestAdditionalCoordinateMetadata:
    def test_error_coord_terminology(self):
        add_coords_cols = {"category_name": ["cat_name", "category"]}
        coords_cols = {"category": "cat_code"}
        coords_terminologies = {"category_name": "IPCC2006_name"}

        with pytest.raises(
            ValueError,
            match="Additional coordinate category_name has terminology definition. "
            "This is currently not supported by PRIMAP2.",
        ):
            additional_coordinate_metadata(
                coords_cols=coords_cols,
                add_coords_cols=add_coords_cols,
                coords_terminologies=coords_terminologies,
            )

    def test_error_coord_not_present(self):
        add_coords_cols = {"category_name": ["cat_name", "category"]}
        coords_cols = {"entity": "entity"}
        coords_terminologies = {}

        with pytest.raises(
            ValueError,
            match="Additional coordinate category_name refers to unknown "
            "coordinate category",
        ):
            additional_coordinate_metadata(
                coords_cols=coords_cols,
                add_coords_cols=add_coords_cols,
                coords_terminologies=coords_terminologies,
            )


class TestRegression:
    def test_read_published_data(self):
        actual = pm2io.from_interchange_format(
            pm2io.read_interchange_format(
                DATA_PATH / "Guetschow-et-al-2021-PRIMAP-crf96_2021-v1"
            )
        )
        expected = primap2.open_dataset(
            DATA_PATH / "Guetschow-et-al-2021-PRIMAP-crf96_2021-v1.nc"
        )
        assert_ds_aligned_equal(actual, expected, equal_nan=True)


# functions that still need individual testing
# dates_to_dimension(ds: xr.Dataset, time_format: str = "%Y") -> xr.DataArray:
# harmonize_units
