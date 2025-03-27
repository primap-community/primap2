#!/usr/bin/env python
"""Tests for csg/_wrapper.py"""

from datetime import datetime
from pathlib import Path

import pandas as pd

import primap2.csg
from primap2.csg import create_composite_source
from primap2.csg._wrapper import create_time_index, set_priority_coords
from primap2.tests.utils import assert_ds_aligned_equal

DATA_PATH = Path(__file__).parent.parent / "data"


def test_set_priority_coords(minimal_ds):
    prio_coords = {"scenario": {"value": "HISTORY", "terminology": "PRIMAP"}}

    prio_coord_ds = set_priority_coords(minimal_ds, prio_coords)

    assert "scenario (PRIMAP)" in prio_coord_ds.coords
    assert prio_coord_ds.coords["scenario (PRIMAP)"].values == ["HISTORY"]


def test_create_time_index():
    start = "1990"
    end = "2000"
    start_dt = datetime.strptime(start, "%Y")
    end_dt = datetime.strptime(end, "%Y")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    expected = pd.date_range(start=start, end=end, freq="YS")

    # string tuple
    pd.testing.assert_index_equal(create_time_index((start, end)), expected)

    # datatime tuple
    pd.testing.assert_index_equal(create_time_index((start_dt, end_dt)), expected)

    # timestamp tuple
    pd.testing.assert_index_equal(create_time_index((start_ts, end_ts)), expected)

    # mixed tuple
    pd.testing.assert_index_equal(create_time_index((start, end_dt)), expected)

    # DatetimeIndex returned unchanged
    pd.testing.assert_index_equal(create_time_index(expected), expected)


def test_create_composite_source():
    cat_terminology = "IPCC2006_PRIMAP"

    main_categories = ["1.A", "1.B.2", "2.A", "M.AG.ELV", "M.LULUCF", "4"]
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
        {"source": "UNFCCC NAI, 240223"},
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

    input_data = primap2.open_dataset(DATA_PATH / "primap2_test_data_v2.5.1_final.nc")

    # we use source as priority dimension, everything else are fixed coordinates.
    # we have one country-specific exception for each country in the prioritization
    # that's likely a bit more than realistic, but let's aim high
    priority_definition = primap2.csg.PriorityDefinition(
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

    result = create_composite_source(
        input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        result_prio_coords=result_prio_coords,
        limit_coords=test_limit_coords,
        time_range=test_time_range,
        progress_bar=None,
        metadata=metadata,
    )

    # remove processing info as following functions can't deal with it yet
    # in this case to_netcdf can't deal with the None values in processing info
    result = result.pr.remove_processing_info()

    # assert results
    # load comparison data
    comp_filename = "PRIMAP-csg-test.nc"
    file_to_load = DATA_PATH / comp_filename
    data_comp = primap2.open_dataset(file_to_load)

    assert_ds_aligned_equal(data_comp, result, equal_nan=True)
