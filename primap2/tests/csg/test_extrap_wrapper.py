"""Tests for csg/_extrapolation_wrapper.py"""

from pathlib import Path

import numpy as np

import primap2.csg
from primap2.csg._extrapolation_wrapper import _csg_extrapolation_prepare_data
from primap2.tests.utils import assert_aligned_equal

DATA_PATH = Path(__file__).parent.parent / "data"


def test_convert_priority_def():
    cat_terminology = "IPCC2006_PRIMAP"

    main_categories = ["1.A", "1.B.2", "2.A", "M.AG.ELV", "M.LULUCF", "4"]

    priorities_extrap = [
        {"source": "CRF 2023, 240108"},
        {
            "source": "UNFCCC NAI, 240223",
            "mapping": {
                "area (ISO3)": {
                    "NORAM": ["USA", "CAN"],
                }
            },
        },
        {
            "source": "CDIAC 2023, HISTORY",
            f"category ({cat_terminology})": ["1.A", "2.A"],
            "entity": "CO2",
            "mapping": {
                "area (ISO3)": {
                    "NORAM": ["USA", "CAN"],
                }
            },
        },
    ]
    priorities = [
        {"source": "CRF 2023, 240108"},
        {
            "source": "UNFCCC NAI, 240223",
        },
        {
            "source": "CDIAC 2023, HISTORY",
            f"category ({cat_terminology})": ["1.A", "2.A"],
            "entity": "CO2",
        },
    ]
    extrap_priority_definition = primap2.csg.ExtrapPriorityDefinition(
        priority_dimensions=["source"],
        priorities=priorities_extrap,
        exclude_result=[
            {
                "entity": ["CO2", "CH4", "N2O"],
                f"category ({cat_terminology})": main_categories,
            },
        ],
    )

    priority_definition = extrap_priority_definition.to_priority_definition()

    assert isinstance(priority_definition, primap2.csg.PriorityDefinition)

    assert priority_definition.priorities == priorities

    assert priority_definition.priority_dimensions == extrap_priority_definition.priority_dimensions

    assert priority_definition.exclude_result == extrap_priority_definition.exclude_result


def test_extrap_data_prep(caplog):
    cat_terminology = "IPCC2006_PRIMAP"

    priorities = [
        # {"source": "CRF 2023, 240108"},
        {
            "source": "UNFCCC NAI, 240223",
            "mapping": {
                f"category ({cat_terminology})": {
                    "2": ["2.A", "2.B"],
                }
            },
            "variable": ["CO2", "CH4"],
        },
        {
            "source": "CRF 2023, 240108",
            "mapping": {
                f"category ({cat_terminology})": {
                    "2": ["2.A", "2.B"],
                }
            },
            "variable": ["CO2", "CH4"],
        },
        {
            "source": "EDGAR 7.0, HISTORY",
            "mapping": {
                f"category ({cat_terminology})": {
                    "2": ["2.B"],
                }
            },
            "entity": ["HFCS"],
        },
        {
            "source": "CDIAC 2023, HISTORY",
            f"category ({cat_terminology})": ["1.A", "2.A"],
            "entity": "CO2",
        },
    ]

    input_data = primap2.open_dataset(DATA_PATH / "primap2_test_data_v2.5.1_final.nc")

    # remove 2.A and 2.B data for UNFCCC NAI
    input_data["CO2"].pr.loc[
        {"source": "UNFCCC NAI, 240223", f"category ({cat_terminology})": ["2.A", "2.B"]}
    ] = (
        input_data["CO2"].pr.loc[
            {"source": "UNFCCC NAI, 240223", f"category ({cat_terminology})": ["2.A", "2.B"]}
        ]
        * np.nan
    )

    # remove 2.A and 2.B data for CRF in CO2 and CH4
    for var in ["CO2", "CH4"]:
        input_data[var].pr.loc[
            {"source": "CRF 2023, 240108", f"category ({cat_terminology})": ["2.A", "2.B"]}
        ] = (
            input_data[var].pr.loc[
                {"source": "CRF 2023, 240108", f"category ({cat_terminology})": ["2.A", "2.B"]}
            ]
            * np.nan
        )

    # remove 2.B HFC data for EDGAR
    for var in ["HFCS (AR5GWP100)", "HFCS (AR6GWP100)"]:
        input_data[var].pr.loc[
            {"source": "EDGAR 7.0, HISTORY", f"category ({cat_terminology})": ["2.B"]}
        ] = (
            input_data[var].pr.loc[
                {"source": "EDGAR 7.0, HISTORY", f"category ({cat_terminology})": ["2.B"]}
            ]
            * np.nan
        )

    priority_definition_extrap = primap2.csg.ExtrapPriorityDefinition(
        priority_dimensions=["source"],
        priorities=priorities,
    )

    data_proc = _csg_extrapolation_prepare_data(input_data, priority_definition_extrap)

    # for CO2 UNFCCC NAI we set data to nan in 2.A and 2.B, so we should have the
    # category 2 data now
    data_2 = input_data["CO2"].pr.loc[{"source": "UNFCCC NAI, 240223", "category": "2"}]
    data_2A = data_proc["CO2"].pr.loc[{"source": "UNFCCC NAI, 240223", "category": "2.A"}]
    data_2B = data_proc["CO2"].pr.loc[{"source": "UNFCCC NAI, 240223", "category": "2.B"}]
    data_2_proc = data_proc["CO2"].pr.loc[{"source": "UNFCCC NAI, 240223", "category": "2"}]
    assert_aligned_equal(data_2, data_2A, equal_nan=True)
    assert_aligned_equal(data_2, data_2B, equal_nan=True)
    # make sure the parent time-series has not changed
    assert_aligned_equal(data_2, data_2_proc, equal_nan=True)

    # for CH4 we have not done so and should have the original data
    data_2A_input = input_data["CH4"].pr.loc[{"source": "UNFCCC NAI, 240223", "category": "2.A"}]
    data_2A_proc = data_proc["CH4"].pr.loc[{"source": "UNFCCC NAI, 240223", "category": "2.A"}]
    assert_aligned_equal(data_2A_input, data_2A_proc, equal_nan=True)

    # for CO2, CH4 CRF we set data to nan in 2.A and 2.B, so we should have the
    # category 2 data now
    variables = ["CH4", "CO2"]
    for var in variables:
        data_2 = input_data[var].pr.loc[{"source": "CRF 2023, 240108", "category": "2"}]
        data_2A = data_proc[var].pr.loc[{"source": "CRF 2023, 240108", "category": "2.A"}]
        data_2B = data_proc[var].pr.loc[{"source": "CRF 2023, 240108", "category": "2.B"}]
        data_2_proc = data_proc[var].pr.loc[{"source": "CRF 2023, 240108", "category": "2"}]
        assert_aligned_equal(data_2, data_2A, equal_nan=True)
        assert_aligned_equal(data_2, data_2B, equal_nan=True)
        # make sure the parent time-series has not changed
    assert_aligned_equal(data_2, data_2_proc, equal_nan=True)

    # for HFCS from EDGAR we set data to nan in 2.B, so we should have the
    # category 2 data now
    variables = ["HFCS (AR5GWP100)", "HFCS (AR6GWP100)"]
    for var in variables:
        data_2 = input_data[var].pr.loc[{"source": "EDGAR 7.0, HISTORY", "category": "2"}]
        data_2B = data_proc[var].pr.loc[{"source": "EDGAR 7.0, HISTORY", "category": "2.B"}]
        data_2_proc = data_proc[var].pr.loc[{"source": "EDGAR 7.0, HISTORY", "category": "2"}]
        assert_aligned_equal(data_2, data_2B, equal_nan=True)
        # make sure the parent time-series has not changed
    assert_aligned_equal(data_2, data_2_proc, equal_nan=True)

    # check that all variables are still present
    assert list(input_data.data_vars) == list(data_proc.data_vars)
