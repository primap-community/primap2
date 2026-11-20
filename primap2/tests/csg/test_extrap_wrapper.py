"""Tests for csg/_extrapolation_wrapper.py"""

import logging
from pathlib import Path

import numpy as np

import primap2.csg
from primap2.csg._extrapolation_wrapper import (
    _csg_extrapolation_prepare_data,
    create_extrapolated_source,
)
from primap2.tests.utils import assert_aligned_equal, assert_ds_aligned_equal

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
        {"source": "CRF 2023, 240108"},
        {
            "source": "UNFCCC NAI, 240223",
            "mapping": {
                f"category ({cat_terminology})": {
                    "2": ["2.A", "2.B"],
                }
            },
            "variable": ["CO2"],
        },
        {
            "source": "UNFCCC NAI, 240223",
            "variable": ["CH4"],
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


def test_create_extrapolated_source(caplog):
    cat_terminology = "IPCC2006_PRIMAP"

    main_categories = ["1.A", "1.B.2", "2.A", "2.B", "M.AG.ELV", "M.LULUCF", "4"]
    FGAS_categories = ["2"]

    native_entities = ["CO2", "CH4", "N2O", "SF6"]
    GWP_entities = ["HFCS"]
    GWPs = ["AR6GWP100"]
    GWP_variables = [f"{entity} ({GWP})" for entity in GWP_entities for GWP in GWPs]
    FGAS_entities = ["SF6", *GWP_entities]
    FGAS_variables = ["SF6", *GWP_variables]
    variables = native_entities + GWP_variables
    # priority
    priorities = [
        {"source": "CRF 2023, 240108"},
        {
            "source": "CRF 2023, 240108",
            "mapping": {
                f"category ({cat_terminology})": {
                    "2": ["2.A", "2.B"],
                },
            },
            "entity": ["CO2"],
            "area (ISO3)": "DEU",
        },
        {
            "source": "UNFCCC NAI, 240223",
        },
        {
            "source": "UNFCCC NAI, 240223",
            "mapping": {
                f"category ({cat_terminology})": {
                    "2": ["2.A", "2.B"],
                },
            },
            "entity": ["CO2", "CH4"],
        },
        {
            "source": "CDIAC 2023, HISTORY",
            f"category ({cat_terminology})": ["1.A", "2.A"],
            "entity": "CO2",
        },
        {
            "source": "Andrew cement, HISTORY",
            f"category ({cat_terminology})": ["2.A"],
            "entity": "CO2",
        },
        {
            "source": "EI 2023, HISTORY",
            f"category ({cat_terminology})": ["1.A", "1.B.2"],
            "entity": "CO2",
        },
        {"source": "Houghton, HISTORY", f"category ({cat_terminology})": "M.LULUCF"},
        {"source": "FAOSTAT 2023, HISTORY", f"category ({cat_terminology})": ["M.AG.ELV"]},
        {"source": "EDGAR 8.0, HISTORY", "entity": ["CO2", "CH4", "N2O"]},
        {
            "source": "EDGAR 7.0, HISTORY",
            f"category ({cat_terminology})": FGAS_categories,
            "variable": FGAS_variables,
        },
    ]

    used_sources = [prio["source"] for prio in priorities]
    FGAS_sources = [
        "CRF 2023, 240108",
        "CRF 2022, 230510",
        "UNFCCC NAI, 240223",
        "EDGAR 7.0, HISTORY",
    ]

    result_prio_coords = {
        "source": {"value": "PRIMAP-test"},
        "scenario": {"value": "HISTORY", "terminology": "PRIMAP"},
    }

    metadata = {"references": "test-data", "contact": "test@example.xx"}

    # regenerate using data/generate_primap2_test_data.py, which needs the PRIMAP-hist
    # input data that is not part of this repository
    input_data = primap2.open_dataset(DATA_PATH / "primap2_test_data_v2.5.1_final.nc")

    # remove 2.A and 2.B  CO2, CH4  data for UNFCCC NAI
    for var in ["CO2", "CH4"]:
        input_data[var].pr.loc[
            {"source": ["UNFCCC NAI, 240223"], f"category ({cat_terminology})": ["2.A", "2.B"]}
        ] = (
            input_data[var].pr.loc[
                {"source": ["UNFCCC NAI, 240223"], f"category ({cat_terminology})": ["2.A", "2.B"]}
            ]
            * np.nan
        )

    # remove CO2 data for 2.A, 2.B for DEU
    input_data["CO2"].pr.loc[
        {
            "source": "CRF 2023, 240108",
            f"category ({cat_terminology})": ["2.A", "2.B"],
            "area (ISO3)": "DEU",
        }
    ] = (
        input_data["CO2"].pr.loc[
            {
                "source": "CRF 2023, 240108",
                f"category ({cat_terminology})": ["2.A", "2.B"],
                "area (ISO3)": "DEU",
            }
        ]
        * np.nan
    )

    # we use source as priority dimension, everything else are fixed coordinates.
    # we have one country-specific exception for each country in the prioritization
    # that's likely a bit more than realistic, but let's aim high
    priority_definition = primap2.csg.ExtrapPriorityDefinition(
        priority_dimensions=["source"],
        priorities=priorities,
        exclude_result=[
            {
                "entity": ["CO2", "CH4", "N2O"],
                f"category ({cat_terminology})": FGAS_categories,
            },
            {
                "entity": FGAS_entities,
                f"category ({cat_terminology})": main_categories,
            },
        ],
    )

    strategies_FGAS = [
        (
            {
                "source": FGAS_sources,
                "entity": FGAS_entities,
                f"category ({cat_terminology})": FGAS_categories,
            },
            primap2.csg.GlobalLSStrategy(),
        ),
        (
            {
                "source": FGAS_sources,
                "entity": FGAS_entities,
                f"category ({cat_terminology})": FGAS_categories,
            },
            primap2.csg.SubstitutionStrategy(),
        ),
    ]

    strategies_CO2CH4N2O = [
        (
            {
                "source": used_sources,
                "entity": ["CO2", "CH4", "N2O"],
                f"category ({cat_terminology})": main_categories,
            },
            primap2.csg.GlobalLSStrategy(),
        ),
        (
            {
                "source": used_sources,
                "entity": ["CO2", "CH4", "N2O"],
                f"category ({cat_terminology})": main_categories,
            },
            primap2.csg.SubstitutionStrategy(),
        ),
    ]

    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=strategies_CO2CH4N2O + strategies_FGAS
    )

    test_time_range = ("1962", "2022")  # cut a few years to make sure that works
    # test_limit_coords = {'entity': ['CO2', 'CH4', 'N2O']}
    test_limit_coords = {
        "variable": variables,
        "category": main_categories + FGAS_categories,
        "source": used_sources,
    }

    result = create_extrapolated_source(
        input_data,
        extrap_priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        result_prio_coords=result_prio_coords,
        limit_coords=test_limit_coords,
        time_range=test_time_range,
        progress_bar=None,
        metadata=metadata,
    )

    # create_composite_source runs ensure_valid on the result, which only logs most
    # problems instead of raising, so check that nothing was reported
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]

    # remove processing info, the comparison data was created without it
    result = result.pr.remove_processing_info()

    # assert results
    # load comparison data
    comp_filename = "PRIMAP-csg-extrap-test.nc"
    file_to_load = DATA_PATH / comp_filename
    data_comp = primap2.open_dataset(file_to_load)

    assert_ds_aligned_equal(data_comp, result, equal_nan=True)
