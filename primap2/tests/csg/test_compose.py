"""Tests for csg/_compose.py"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2.csg
import primap2.csg._compose
from primap2.tests.csg.utils import get_single_ts


def assert_copied_from_input_data(
    filtered_result: xr.DataArray,
    filtered_initial: xr.DataArray,
    common_filter: dict[str, str],
):
    """Assert that timeseries in the result were copied from the input data."""
    expected = (
        filtered_initial.pr.loc[common_filter]
        .drop_vars("scenario (FAOSTAT)")
        .drop_vars("source")
        .expand_dims(dim={"source": ["composed"]})
    )
    result = filtered_result.pr.loc[common_filter]
    xr.testing.assert_identical(result, expected)


def test_compose_simple(opulent_ds):
    """A test with relatively complex input data but simple rules.

    The input data has many dimensions and we use two as priority dimensions. There
    are some NaN data points in one source.

    The input rules use two sources for all entities, plus one additional for CH4. For
    every entity, we use the simple SubstitutionStrategy for all sources.
    """
    input_data = opulent_ds
    input_data = input_data.drop_vars(["population", "SF6 (SARGWP100)"])
    input_data["CO2"].loc[{"source": "RAND2020", "time": ["2000", "2001"]}] = (
        np.nan * primap2.ureg("Mt CO2 / year")
    )
    # we now have dimensions time, area (ISO3), category (IPCC 2006), animal (FAOSTAT)
    # product (FAOSTAT), scenario (FAOSTAT), provenance, model, source
    # We have variables (entities): CO2, SF6, CH4
    # We have sources: RAND2020, RAND2021
    # We have scenarios: highpop, lowpop
    # Idea: we use source, scenario as priority dimensions, everything else are fixed
    # coordinates.

    # generally, prefer source RAND2020, scenario lowpop,
    # then use source RAND2021, scenario highpop.
    # however, for Columbia CH4, use source RAND2020, scenario highpop as highest priority
    # (this combination is not used at all otherwise).
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {
                "entity": "CH4",
                "area (ISO3)": "COL",
                "source": "RAND2020",
                "scenario (FAOSTAT)": "highpop",
            },
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
    )
    # we use straight substitution always, but specify it somewhat fancy
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            (
                {"entity": ent},
                primap2.csg.SubstitutionStrategy(),
            )
            for ent in input_data.keys()
        ]
    )

    result = primap2.csg.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )
    # The caller of `compose` is responsible for re-adding priority dimensions
    # if necessary
    result = result.expand_dims(dim={"source": ["composed"]})
    result.pr.ensure_valid()
    assert "CO2" in result.keys()
    assert "Processing of CO2" in result.keys()
    assert_copied_from_input_data(
        result["CH4"],
        input_data["CH4"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "highpop"}],
        {"area": "COL"},
    )
    result_col_proc = (
        result["Processing of CH4"].loc[{"area (ISO3)": "COL"}].values.flat[0]
    )
    assert len(result_col_proc.steps) == 1
    assert result_col_proc.steps[0].time == "all"
    assert result_col_proc.steps[0].function == "substitution"
    assert "'source': 'RAND2020'" in result_col_proc.steps[0].description
    assert "'scenario (FAOSTAT)': 'highpop'" in result_col_proc.steps[0].description

    assert_copied_from_input_data(
        result["CH4"],
        input_data["CH4"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {"area": "ARG"},
    )
    result_arg_proc = (
        result["Processing of CH4"].loc[{"area (ISO3)": "ARG"}].values.flat[0]
    )
    assert len(result_arg_proc.steps) == 1
    assert result_arg_proc.steps[0].time == "all"
    assert result_arg_proc.steps[0].function == "substitution"
    assert (
        result_arg_proc.steps[0].source
        == "{'source': 'RAND2020', 'scenario (FAOSTAT)': 'lowpop'}"
    )

    assert_copied_from_input_data(
        result["CO2"],
        input_data["CO2"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {"area": "COL", "time": slice("2002", None)},
    )
    result_col_co2_proc = (
        result["Processing of CO2"].loc[{"area (ISO3)": "COL"}].values.flat[0]
    )
    assert len(result_col_co2_proc.steps) == 2
    assert result_col_co2_proc.steps[0].function == "substitution"
    np.testing.assert_array_equal(
        result_col_co2_proc.steps[1].time,
        np.array(["2000", "2001"], dtype=np.datetime64),
    )
    assert result_col_co2_proc.steps[1].function == "substitution"
    assert "'source': 'RAND2020'" in result_col_co2_proc.steps[0].description
    assert "'scenario (FAOSTAT)': 'lowpop'" in result_col_co2_proc.steps[0].description


def test_compose_null_strategy(opulent_ds):
    """In this test, we override one entity using the NullStrategy()."""
    input_data = opulent_ds.drop_vars(["population", "SF6 (SARGWP100)"])

    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
    )
    # for CH4, we want to skip the 2020 source, for SF6 we want to skip all sources
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"entity": "CH4", "source": "RAND2020"}, primap2.csg.NullStrategy()),
            ({"entity": "SF6"}, primap2.csg.NullStrategy()),
            ({}, primap2.csg.SubstitutionStrategy()),
        ]
    )

    result = primap2.csg.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )
    # The caller of `compose` is responsible for re-adding priority dimensions
    # if necessary
    result = result.expand_dims(dim={"source": ["composed"]})
    result.pr.ensure_valid()

    assert "CO2" in result.keys()
    assert "Processing of CO2" in result.keys()

    assert_copied_from_input_data(
        result["CO2"],
        input_data["CO2"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {},
    )

    assert_copied_from_input_data(
        result["CH4"],
        input_data["CH4"].loc[{"source": "RAND2021", "scenario (FAOSTAT)": "highpop"}],
        {},
    )

    result_sf6: xr.DataArray = result["SF6"]
    assert result_sf6.isnull().all().all()


def test_compose_strategy_skipping(opulent_ds):
    """In this test, we use a strategy which raises an error and assert that it is
    skipped properly."""
    input_data = opulent_ds.drop_vars(["population", "SF6 (SARGWP100)"])

    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
    )

    # for CH4, we use a strategy which gives up for the RAND2020 source
    class ErroringStrategy:
        type = "erroring"

        def fill(
            self,
            *,
            ts: xr.DataArray,
            fill_ts: xr.DataArray,
            fill_ts_repr: str,
        ) -> tuple[xr.DataArray, list[primap2.ProcessingStepDescription]]:
            raise primap2.csg.StrategyUnableToProcess("no processing")

    # However, we define the substitution strategy as the fallback strategy, so that
    # the Substitution strategy is used for everything anyway.
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"entity": "CH4", "source": "RAND2020"}, ErroringStrategy()),
            ({}, primap2.csg.SubstitutionStrategy()),
        ]
    )

    result = primap2.csg.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )
    # The caller of `compose` is responsible for re-adding priority dimensions
    # if necessary
    result = result.expand_dims(dim={"source": ["composed"]})
    result.pr.ensure_valid()

    assert_copied_from_input_data(
        result["CO2"],
        input_data["CO2"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {},
    )
    assert_copied_from_input_data(
        result["CH4"],
        input_data["CH4"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {},
    )


def test_compose_strategy_all_error(opulent_ds):
    input_data = opulent_ds.drop_vars(["population", "SF6 (SARGWP100)"])

    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
    )

    # for CH4, we use a strategy which gives up as the only strategy, which should
    # generate an error
    class ErroringStrategy:
        type = "erroring"

        def fill(
            self,
            *,
            ts: xr.DataArray,
            fill_ts: xr.DataArray,
            fill_ts_repr: str,
        ) -> tuple[xr.DataArray, list[primap2.ProcessingStepDescription]]:
            raise primap2.csg.StrategyUnableToProcess("no processing")

    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"entity": "CH4"}, ErroringStrategy()),
            ({"entity": ["CO2", "SF6"]}, primap2.csg.SubstitutionStrategy()),
        ]
    )

    with pytest.raises(ValueError, match="No configured strategy was able to process"):
        primap2.csg.compose(
            input_data=input_data,
            priority_definition=priority_definition,
            strategy_definition=strategy_definition,
        )


def test_compose_identity_strategy(opulent_ds):
    """Using the identity strategy, we do not use a specific source for CH4."""
    input_data = opulent_ds.drop_vars(["population", "SF6 (SARGWP100)", "SF6"]).pr.loc[
        {"animal": ["cow"], "product": ["milk"], "category": ["0", "1"]}
    ]

    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
    )

    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"entity": "CH4", "source": "RAND2020"}, primap2.csg.IdentityStrategy()),
            ({}, primap2.csg.SubstitutionStrategy()),
        ]
    )

    result = primap2.csg.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )
    # The caller of `compose` is responsible for re-adding priority dimensions
    # if necessary
    result = result.expand_dims(dim={"source": ["composed"]})
    result.pr.ensure_valid()

    assert_copied_from_input_data(
        result["CO2"],
        input_data["CO2"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {},
    )
    assert_copied_from_input_data(
        result["CH4"],
        input_data["CH4"].loc[{"source": "RAND2021", "scenario (FAOSTAT)": "highpop"}],
        {},
    )

    tpd: primap2.TimeseriesProcessingDescription = (
        result["Processing of CH4"].pr.loc[{"area": "COL", "category": "0"}].item()
    )
    assert len(tpd.steps) == 2
    assert tpd.steps[0].function == "identity"
    assert (
        tpd.steps[0].source == "{'source': 'RAND2020', 'scenario (FAOSTAT)': 'lowpop'}"
    )
    assert tpd.steps[1].function == "substitution"


def test_compose_pbar(opulent_ds):
    input_data = opulent_ds.drop_vars(["population", "SF6 (SARGWP100)"])
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            (
                {},
                primap2.csg.SubstitutionStrategy(),
            )
        ]
    )

    result = primap2.csg.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        progress_bar=None,
    )
    assert "CO2" in result.keys()


def test_compose_sec_cats_missing(opulent_ds):
    input_data = opulent_ds.drop_vars(["population", "SF6 (SARGWP100)"])
    input_data.attrs["sec_cats"].remove("product (FAOSTAT)")
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source", "scenario (FAOSTAT)", "product (FAOSTAT)"],
        priorities=[
            {
                "source": "RAND2020",
                "scenario (FAOSTAT)": "lowpop",
                "product (FAOSTAT)": "milk",
            },
            {
                "source": "RAND2021",
                "scenario (FAOSTAT)": "highpop",
                "product (FAOSTAT)": "milk",
            },
        ],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            (
                {},
                primap2.csg.SubstitutionStrategy(),
            )
        ]
    )
    primap2.csg.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        progress_bar=None,
    )


def test_compose_timeseries_trivial():
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"], priorities=[{"source": "A"}, {"source": "B"}]
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"source": "B"}, primap2.csg.SubstitutionStrategy()),
            ({"source": "A"}, primap2.csg.SubstitutionStrategy()),
        ]
    )

    time = pd.date_range("1850-01-01", "2022-01-01", freq="YS")
    anp = np.linspace(0.0, 1.0, len(time))
    anp[0] = np.nan
    anp[1] = np.nan
    da_a = get_single_ts(
        coords={"source": "A", "category": "1", "area (ISO3)": "MEX"},
        data=anp,
        time=time,
    )
    bnp = np.linspace(1000.0, 2000.0, len(time))
    bnp[1] = np.nan
    bnp[2] = np.nan
    da_b = get_single_ts(
        coords={"source": "B", "category": "1", "area (ISO3)": "MEX"},
        data=bnp,
        time=time,
    )

    input_data = xr.concat((da_a, da_b), dim="source", join="exact")

    result_ts, result_description = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )

    enp = np.linspace(0.0, 1.0, len(time))  # same as A
    enp[0] = 1000.0  # filled from B
    enp[1] = np.nan  # not filled, NaN in A and B
    expected_ts = get_single_ts(
        data=enp, coords={"category": "1", "area (ISO3)": "MEX"}, time=time
    )

    xr.testing.assert_identical(result_ts, expected_ts)

    assert len(result_description.steps) == 2
    assert result_description.steps[0].time[0] == np.datetime64("1852-01-01")
    assert result_description.steps[0].function == "substitution"
    assert (
        result_description.steps[0].description
        == "substituted with corresponding values from 'A'"
    )
    assert len(result_description.steps[1].time) == 1
    assert result_description.steps[1].time[0] == np.datetime64("1850-01-01")
    assert result_description.steps[1].function == "substitution"
    assert (
        result_description.steps[1].description
        == "substituted with corresponding values from 'B'"
    )


def test_compose_timeseries_no_match(caplog):
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"],
        priorities=[{"source": "C"}, {"source": "A"}, {"source": "B"}],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"source": "A"}, primap2.csg.SubstitutionStrategy()),
            ({"source": "B"}, primap2.csg.SubstitutionStrategy()),
        ]
    )
    da_a = get_single_ts(
        coords={"source": "A", "category": "1", "area (ISO3)": "MEX"},
    )
    da_b = get_single_ts(
        coords={"source": "B", "category": "1", "area (ISO3)": "MEX"},
    )
    input_data = xr.concat((da_a, da_b), dim="source", join="exact")

    primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )

    assert "selector={'source': 'C'} matched no input_data, skipping." in caplog.text


def test_compose_timeseries_all_null():
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"],
        priorities=[{"source": "A"}, {"source": "B"}],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"source": "A"}, primap2.csg.SubstitutionStrategy()),
            ({"source": "B"}, primap2.csg.SubstitutionStrategy()),
        ]
    )
    da_a = get_single_ts(
        coords={"source": "A", "category": "1", "area (ISO3)": "MEX"},
    )
    da_a[0] = np.nan
    da_b = get_single_ts(
        data=np.nan,
        coords={"source": "B", "category": "1", "area (ISO3)": "MEX"},
    )
    input_data = xr.concat((da_a, da_b), dim="source", join="exact")

    result, result_description = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )

    print(result_description)
    assert result_description.steps[1].description == "'B' is fully NaN, skipped"


def test_compose_timeseries_priorities_wrong():
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"],
        priorities=[{"source": "C"}],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[({"source": "C"}, primap2.csg.SubstitutionStrategy())]
    )
    da_a = get_single_ts(
        coords={"source": "A", "category": "1", "area (ISO3)": "MEX"},
    )
    da_b = get_single_ts(
        data=np.nan,
        coords={"source": "B", "category": "1", "area (ISO3)": "MEX"},
    )
    input_data = xr.concat((da_a, da_b), dim="source", join="exact")

    with pytest.raises(ValueError):
        primap2.csg._compose.compose_timeseries(
            input_data=input_data,
            priority_definition=priority_definition,
            strategy_definition=strategy_definition,
        )


def test_priority_coordinates_repr():
    assert (
        primap2.csg._compose.priority_coordinates_repr(
            fill_ts=get_single_ts(coords={"source": "A"}),
            priority_dimensions=["source"],
        )
        == "'A'"
    )
    assert (
        primap2.csg._compose.priority_coordinates_repr(
            fill_ts=get_single_ts(coords={"source": "A", "scenario": "S"}),
            priority_dimensions=["source", "scenario"],
        )
        == "{'source': 'A', 'scenario': 'S'}"
    )
    assert (
        primap2.csg._compose.priority_coordinates_repr(
            fill_ts=get_single_ts(coords={"source": "A", "scenario": "S"}),
            priority_dimensions=["scenario"],
        )
        == "'S'"
    )
