"""Tests for csg/_compose.py"""

import attrs
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2.csg
from primap2.tests.csg.utils import get_single_ts


def nodes_of(da: xr.DataArray, **selector) -> list:
    """The processing steps over time of the first timeseries of da matching selector."""
    da = da.pr.loc[selector]
    return da.isel({dim: 0 for dim in da.dims if dim != "time"}).transpose("time").values.tolist()


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
    input_data = input_data.drop_vars(["population"]).pr.loc[{"category": ["0", "1"]}]
    input_data["CO2"].loc[{"source": "RAND2020", "time": ["2000", "2001"]}] = np.nan * primap2.ureg(
        "Mt CO2 / year"
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
            for ent in input_data
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
    assert "CO2" in result
    assert "Processing of CO2" in result
    assert_copied_from_input_data(
        result["CH4"],
        input_data["CH4"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "highpop"}],
        {"area": "COL"},
    )
    result_col_proc = nodes_of(result["Processing of CH4"], **{"area (ISO3)": "COL"})
    assert len(result_col_proc) == 21
    for node in result_col_proc:
        # untracked input: the parent is a leaf naming the source
        (leaf,) = node.parents
        assert leaf.parents == ()
        assert leaf.time == node.time
        assert leaf.description.function == "compose"
        assert leaf.description.source == "{'source': 'RAND2020', 'scenario (FAOSTAT)': 'highpop'}"
        assert node.description.function == "substitution"
        assert "'source': 'RAND2020'" in node.description.description
        assert "'scenario (FAOSTAT)': 'highpop'" in node.description.description
    assert [node.time for node in result_col_proc] == list(result["time"].values)
    # one description for all data points set by the step
    assert len({id(node.description) for node in result_col_proc}) == 1

    assert_copied_from_input_data(
        result["CH4"],
        input_data["CH4"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {"area": "ARG"},
    )
    result_arg_proc = nodes_of(result["Processing of CH4"], **{"area (ISO3)": "ARG"})
    assert len(result_arg_proc) == 21
    for node in result_arg_proc:
        assert node.description.function == "substitution"
        assert node.description.source is None

    assert_copied_from_input_data(
        result["CO2"],
        input_data["CO2"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {"area": "COL", "time": slice("2002", None)},
    )
    result_col_co2_proc = nodes_of(result["Processing of CO2"], **{"area (ISO3)": "COL"})
    assert len(result_col_co2_proc) == 21
    for i, node in enumerate(result_col_co2_proc):
        assert node.description.function == "substitution"
        # the first two years are missing in the highest priority source
        scenario = "highpop" if i < 2 else "lowpop"
        source = "RAND2021" if i < 2 else "RAND2020"
        assert f"'source': '{source}'" in node.description.description
        assert f"'scenario (FAOSTAT)': '{scenario}'" in node.description.description


def test_compose_exclude_result(opulent_ds):
    """In this test, we exclude parts of the result from processing."""
    input_data = opulent_ds.drop_vars(["population"]).pr.loc[
        {"animal": ["cow"], "product": ["milk"], "category": ["0", "1"]}
    ]

    # for CH4, we want to exclude the 1 category
    # we want to exclude SF6 fully.
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
        exclude_result=[
            {"entity": "CH4", "category (IPCC 2006)": "1"},
            {"entity": "SF6"},
        ],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"entity": ["CO2", "CH4"]}, primap2.csg.SubstitutionStrategy()),
        ],
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

    assert "CO2" in result
    assert "Processing of CO2" in result

    assert_copied_from_input_data(
        result["CO2"],
        input_data["CO2"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {},
    )

    assert_copied_from_input_data(
        result["CH4"],
        input_data["CH4"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {"category": "0"},
    )

    assert result["CH4"].pr.loc[{"category": "1"}].isnull().all().all()

    result_sf6: xr.DataArray = result["SF6"]
    assert result_sf6.isnull().all().all()


def test_compose_invalid_strategy_definition(opulent_ds):
    """We use an invalid strategy definition and verify it raises an error."""
    input_data = opulent_ds.drop_vars(["population"]).pr.loc[
        {
            "animal": ["cow"],
            "product": ["milk"],
            "category": ["0", "1"],
            "area": ["COL"],
        }
    ]
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"], priorities=[{"source": "RAND2020"}]
    )
    valid_strategy_definition = primap2.csg.StrategyDefinition(
        [({}, primap2.csg.SubstitutionStrategy())],
    )
    primap2.csg.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=valid_strategy_definition,
    )
    invalid_strategy_definition = primap2.csg.StrategyDefinition(
        [
            ({"notadim": "RAND2020"}, primap2.csg.SubstitutionStrategy()),
            ({}, primap2.csg.SubstitutionStrategy()),
        ],
    )
    with pytest.raises(ValueError, match="is not a valid dimension"):
        primap2.csg.compose(
            input_data=input_data,
            priority_definition=priority_definition,
            strategy_definition=invalid_strategy_definition,
        )


def test_compose_strategy_skipping(opulent_ds):
    """In this test, we use a strategy which raises an error and assert that it is
    skipped properly.
    """
    input_data = opulent_ds.drop_vars(["population"]).pr.loc[
        {"animal": ["cow"], "product": ["milk"], "category": ["0", "1"]}
    ]

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
    input_data = opulent_ds.drop_vars(["population"]).pr.loc[
        {"animal": ["cow"], "product": ["milk"], "category": ["0", "1"]}
    ]

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


def test_compose_skip_source(opulent_ds):
    """We do not use a specific source for CH4 category 0."""
    input_data = opulent_ds.drop_vars(["population", "SF6"]).pr.loc[
        {"animal": ["cow"], "product": ["milk"], "category": ["0", "1"]}
    ]

    # we want to only exclude RAND2020, lowpop, CH4, 0
    # without excluding       RAND2020, lowpop, CO2, 0
    # or                      RAND2020, lowpop, CH4, 1
    # but each dimension is evaluated individually, so we have to select three
    # times, first everything where entity!=CH4 and category!=0, then
    # entity=CH4 but category!=0 and entity!=CH4 but category=0.
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
        exclude_input=[
            {
                "source": "RAND2020",
                "scenario (FAOSTAT)": "lowpop",
                "entity": "CH4",
                "category (IPCC 2006)": "0",
            }
        ],
    )

    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
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
        {"category": "0"},
    )
    assert_copied_from_input_data(
        result["CH4"],
        input_data["CH4"].loc[{"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"}],
        {"category": "1"},
    )

    nodes = nodes_of(result["Processing of CH4"], area="COL", category="0")
    for node in nodes:
        assert len(node.history()) == 2
        assert node.description.function == "substitution"
        assert node.description.source is None


def test_compose_skip_variable(opulent_ds):
    """We do skip processing SF6 (SARGWP100) altogether.

    Excluding by variable matches only the variable of that exact name.
    """
    input_data = (
        opulent_ds.drop_vars(["population"])
        .pr.loc[
            {
                "animal": ["cow"],
                "product": ["milk"],
                "category": ["0", "1"],
                "area": ["COL"],
            }
        ]
        .pr.convert_to_gwp(gwp_context="SARGWP100", units="CO2 Gg / year")
    )

    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"],
        priorities=[
            {"source": "RAND2020"},
        ],
        exclude_result=[{"variable": "SF6 (SARGWP100)"}],
    )

    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
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

    assert result["SF6 (SARGWP100)"].isnull().all()
    assert not result["CO2 (SARGWP100)"].isnull().all()


def test_compose_skip_entity(opulent_ds):
    """We do skip processing SF6, even though the variable is named differently.

    Excluding by entity matches the variable of that entity, whatever its name is.
    """
    input_data = (
        opulent_ds.drop_vars(["population"])
        .pr.loc[
            {
                "animal": ["cow"],
                "product": ["milk"],
                "category": ["0", "1"],
                "area": ["COL"],
            }
        ]
        .pr.convert_to_gwp(gwp_context="SARGWP100", units="CO2 Gg / year")
    )

    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"],
        priorities=[
            {"source": "RAND2020"},
        ],
        exclude_result=[{"entity": "SF6"}],
    )

    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
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

    assert result["SF6 (SARGWP100)"].isnull().all()
    assert not result["CO2 (SARGWP100)"].isnull().all()


def test_compose_variable_entity(opulent_ds):
    """Test that no strategy is found when specifying variable names for entity"""
    input_data = (
        opulent_ds[["SF6"]]
        .pr.convert_to_gwp(gwp_context="SARGWP100", units="CO2 Gg / year")
        .pr.loc[
            {
                "animal": ["cow"],
                "product": ["milk"],
                "category": ["0", "1"],
                "area": ["COL"],
            }
        ]
    )

    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"],
        priorities=[
            {"source": "RAND2020"},
        ],
    )

    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"entity": ["SF6 (SARGWP100)"]}, primap2.csg.SubstitutionStrategy()),
        ]
    )

    with pytest.raises(ValueError, match="No configured strategy was able to process"):
        primap2.csg.compose(
            input_data=input_data,
            priority_definition=priority_definition,
            strategy_definition=strategy_definition,
        )


def test_compose_pbar(opulent_ds):
    input_data = opulent_ds.drop_vars(["population"]).pr.loc[
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
    assert "CO2" in result


def test_compose_sec_cats_missing(opulent_ds):
    """Compose should also work when a secondary category dimension is missing."""
    input_data = opulent_ds.drop_vars(["population"]).pr.loc[
        {"animal": ["cow"], "category": ["0", "1"]}
    ]
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

    result_ts, result_steps = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )

    enp = np.linspace(0.0, 1.0, len(time))  # same as A
    enp[0] = 1000.0  # filled from B
    enp[1] = np.nan  # not filled, NaN in A and B
    expected_ts = get_single_ts(data=enp, coords={"category": "1", "area (ISO3)": "MEX"}, time=time)

    xr.testing.assert_identical(result_ts, expected_ts)

    assert result_steps.shape == (len(time),)
    # filled from B, untracked input: the parent is a leaf with the source
    (leaf,) = result_steps[0].parents
    assert leaf.parents == ()
    assert leaf.time == time[0]
    assert leaf.description.source == "'B'"
    assert result_steps[0].time == time[0]
    assert result_steps[0].description.function == "substitution"
    assert (
        result_steps[0].description.description == "substituted with corresponding values from 'B'"
    )
    assert result_steps[0].description.source is None
    # no data in A and B
    assert result_steps[1] is None
    for i in range(2, len(time)):
        (leaf,) = result_steps[i].parents
        assert leaf.parents == ()
        assert leaf.description.source == "'A'"
        assert result_steps[i].time == time[i]
        assert result_steps[i].description.source is None
        assert result_steps[i].description is result_steps[2].description
    assert result_steps[2].description.variable == "CH4"
    assert (
        result_steps[2].description.description == "substituted with corresponding values from 'A'"
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


def test_compose_timeseries_all_null(caplog):
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

    _, result_steps = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )

    assert result_steps[0] is None
    assert all(step.description.source is None for step in result_steps[1:])
    assert "'B' is fully NaN, skipped." in caplog.text


def _read_steps(time, description: str, has_data: np.ndarray | None = None):
    """Input processing of a timeseries: a step per time point with data."""
    desc = primap2.ProcessingStepDescription(
        function="read", description=description, source=description
    )
    if has_data is None:
        has_data = np.ones(len(time), dtype=bool)
    steps = np.full(len(time), None, dtype=object)
    for i in np.flatnonzero(has_data):
        steps[i] = primap2.ProcessingStep(desc, time[i])
    return steps


def test_compose_timeseries_input_processing():
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"],
        priorities=[{"source": "A"}, {"source": "B"}],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[({}, primap2.csg.SubstitutionStrategy())]
    )
    da_a = get_single_ts(coords={"source": "A"})
    da_a[0] = np.nan
    da_b = get_single_ts(coords={"source": "B"})
    input_data = xr.concat((da_a, da_b), dim="source", join="exact")
    time = input_data["time"].values
    steps_a = _read_steps(time, "A", input_data.sel(source="A").notnull().values)
    steps_b = _read_steps(time, "B")
    input_processing = xr.DataArray(
        np.stack([steps_a, steps_b]),
        dims=["source", "time"],
        coords={"source": ["A", "B"], "time": time},
    )

    _, result_steps = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        input_processing=input_processing,
    )

    # the input steps are the parents at the same time point
    assert len(result_steps[0].parents) == 1
    assert result_steps[0].parents[0] is steps_b[0]
    assert result_steps[0].description.source is None
    for i in range(1, len(time)):
        assert len(result_steps[i].parents) == 1
        assert result_steps[i].parents[0] is steps_a[i]
        assert result_steps[i].description.source is None
        assert result_steps[i].time == time[i]


def test_compose_timeseries_input_processing_unused(caplog):
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"],
        priorities=[{"source": "A"}, {"source": "B"}],
        exclude_input=[{"source": "A"}],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[({}, primap2.csg.GlobalLSStrategy()), ({}, primap2.csg.SubstitutionStrategy())]
    )
    da_a = get_single_ts(coords={"source": "A"})
    da_b = get_single_ts(coords={"source": "B"})
    da_c = get_single_ts(coords={"source": "C"})
    input_data = xr.concat((da_a, da_b, da_c), dim="source", join="exact")
    time = input_data["time"].values
    steps = [_read_steps(time, source) for source in "ABC"]
    input_processing = xr.DataArray(
        np.stack(steps),
        dims=["source", "time"],
        coords={"source": ["A", "B", "C"], "time": time},
    )

    _, result_steps = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        input_processing=input_processing,
    )

    for i, step in enumerate(result_steps):
        assert step.parents == (steps[1][i],)
    assert "'A' is excluded from processing, skipped." in caplog.text
    assert "Strategy globalLS unable to process 'B' (No overlap" in caplog.text


def _ab_input():
    """Timeseries A (NaN at 0, 1 and 5) and a full B with tracked processing."""
    da_a = get_single_ts(coords={"source": "A"})
    da_a[[0, 1, 5]] = np.nan
    da_a[2:5] = np.array([1.0, 2.0, 3.0])
    da_b = get_single_ts(coords={"source": "B"})
    da_b[:] = np.linspace(10.0, 20.0, len(da_b))
    input_data = xr.concat((da_a, da_b), dim="source", join="exact")
    time = input_data["time"].values
    steps_a = _read_steps(time, "A", input_data.sel(source="A").notnull().values)
    steps_b = _read_steps(time, "B")
    input_processing = xr.DataArray(
        np.stack([steps_a, steps_b]),
        dims=["source", "time"],
        coords={"source": ["A", "B"], "time": time},
    )
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source"], priorities=[{"source": "A"}, {"source": "B"}]
    )
    return input_data, input_processing, priority_definition, steps_a, steps_b


def test_compose_timeseries_global_ls_parents():
    input_data, input_processing, priority_definition, steps_a, steps_b = _ab_input()
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"source": "A"}, primap2.csg.SubstitutionStrategy()),
            ({"source": "B"}, primap2.csg.GlobalLSStrategy()),
        ]
    )

    result_ts, result_steps = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        input_processing=input_processing,
    )

    assert result_ts.notnull().all()
    # the values from A are not changed by the fill
    for i in range(2, 5):
        assert result_steps[i].parents == (steps_a[i],)
    overlap = [i for i in range(len(steps_a)) if i not in (0, 1, 5)]
    # the fit used the overlap of the result so far (A) and B
    expected_common = [result_steps[i] for i in overlap] + [steps_b[i] for i in overlap]
    for i in (0, 1, 5):
        parents = result_steps[i].parents
        assert parents[0] is steps_b[i]
        assert len(parents) == 1 + len(expected_common)
        assert all(x is y for x, y in zip(parents[1:], expected_common, strict=True))
        assert result_steps[i].description.function == "globalLS"
    # the steps of the result from A are the ones in the parents, not their inputs
    assert result_steps[overlap[0]].parents == (steps_a[overlap[0]],)


def test_compose_timeseries_global_ls_used_sources():
    input_data, input_processing, priority_definition, _, _ = _ab_input()
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"source": "A"}, primap2.csg.SubstitutionStrategy()),
            ({"source": "B"}, primap2.csg.GlobalLSStrategy()),
        ]
    )
    result_ts, result_steps = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        input_processing=input_processing,
    )
    time = input_data["time"]
    ds = xr.Dataset(
        {
            "CO2": xr.DataArray(result_ts.values, dims=["time"], coords={"time": time}),
            "Processing of CO2": xr.DataArray(
                result_steps,
                dims=["time"],
                coords={"time": time},
                attrs={"described_variable": "CO2"},
            ),
        }
    )

    used = ds.pr.used_sources()["CO2"]

    assert list(used["used source"].values) == ["A", "B"]
    # where B fills the gaps of A, A is only used through the fit
    expected = np.tile([1, 0], (len(time), 1))
    expected[[0, 1, 5], 1] = 1
    assert (used.values == expected).all()


def test_compose_timeseries_global_ls_untracked():
    input_data, _, priority_definition, _, _ = _ab_input()
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"source": "A"}, primap2.csg.SubstitutionStrategy()),
            ({"source": "B"}, primap2.csg.GlobalLSStrategy()),
        ]
    )

    _, result_steps = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )

    # the fill from B is reachable as leaves of B, also for the fit points
    for i in (0, 1, 5):
        leaves = [step for step in result_steps[i].history() if not step.parents]
        sources = {leaf.description.source for leaf in leaves}
        assert sources == {"'A'", "'B'"}
        b_times = {leaf.time for leaf in leaves if leaf.description.source == "'B'"}
        # the leaves of B are at the changed time point and the fit points
        assert result_steps[i].time in b_times
        assert len(b_times) == 1 + 3 + (len(result_steps) - 6)


@attrs.frozen
class ReplaceStrategy:
    """Replace the value at the 4th time point, declaring the inputs as configured."""

    ts_time_self: bool = False
    fill_ts_time_self: bool = False
    discard: bool = False
    type = "replace"

    def fill(self, *, ts, fill_ts, fill_ts_repr):
        time = ts["time"].values[3:4]
        result = ts.copy()
        result.loc[{"time": time}] = np.nan if self.discard else fill_ts.loc[{"time": time}]
        step = primap2.csg.StrategyStep(
            function="replace",
            description="replace",
            time=time,
            ts_time=time if self.ts_time_self else [],
            fill_ts_time=time if self.fill_ts_time_self else [],
        )
        return result, [step]


@pytest.mark.parametrize(
    ("strategy", "old_value_is_parent"),
    [
        # replaced values drop their history
        (ReplaceStrategy(), False),
        # unless the old value is an input
        (ReplaceStrategy(ts_time_self=True), True),
        # inputs are not duplicated
        (ReplaceStrategy(fill_ts_time_self=True), False),
        (ReplaceStrategy(ts_time_self=True, fill_ts_time_self=True), True),
        # no data, no step
        (ReplaceStrategy(discard=True), None),
    ],
)
def test_compose_timeseries_replaced_values(strategy, old_value_is_parent):
    input_data, input_processing, priority_definition, steps_a, steps_b = _ab_input()
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            ({"source": "A"}, primap2.csg.SubstitutionStrategy()),
            ({"source": "B"}, strategy),
        ]
    )

    _, result_steps = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        input_processing=input_processing,
    )

    # only the 4th value was changed, the others keep their steps
    assert result_steps[0] is None
    assert result_steps[2].parents == (steps_a[2],)
    if old_value_is_parent is None:
        assert result_steps[3] is None
        return
    parents = result_steps[3].parents
    assert parents[0] is steps_b[3]
    if old_value_is_parent:
        assert len(parents) == 2
        assert parents[1].description.function == "substitution"
        assert parents[1].parents == (steps_a[3],)
    else:
        assert len(parents) == 1


def test_compose_input_processing(opulent_processing_ds):
    input_data = opulent_processing_ds.drop_vars(["population", "Processing of population"])
    input_data = input_data.pr.loc[{"category": ["0", "1"]}]
    input_data["CO2"].loc[{"source": "RAND2020", "time": ["2000", "2001"]}] = np.nan * primap2.ureg(
        "Mt CO2 / year"
    )
    input_data["Processing of CO2"].loc[{"source": "RAND2020", "time": ["2000", "2001"]}] = None
    priority_definition = primap2.csg.PriorityDefinition(
        priority_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[({}, primap2.csg.SubstitutionStrategy())]
    )

    result = primap2.csg.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        progress_bar=None,
    )

    result.expand_dims(dim={"source": ["composed"]}).pr.ensure_valid()
    assert result["Processing of CO2"].dims == result["CO2"].dims
    fixed = {
        dim: input_data["CO2"][dim].values[0].item() for dim in result["CO2"].dims if dim != "time"
    }

    def input_nodes(variable, source, scenario):
        da = input_data[f"Processing of {variable}"].loc[
            {"source": source, "scenario (FAOSTAT)": scenario, **fixed}
        ]
        return da.transpose("time").values

    lowpop = input_nodes("CO2", "RAND2020", "lowpop")
    highpop = input_nodes("CO2", "RAND2021", "highpop")
    co2 = result["Processing of CO2"].loc[fixed].transpose("time").values
    for i, step in enumerate(co2):
        # the first two years are missing in the highest priority source
        expected = highpop[i] if i < 2 else lowpop[i]
        assert len(step.parents) == 1
        assert step.parents[0] is expected
        assert step.time == result["time"].values[i]
        assert step.description.function == "substitution"
        assert step.description.variable == "CO2"
        assert step.description.coords == fixed
    assert [step.description.function for step in co2[0].history()] == ["random", "substitution"]
    ch4 = result["Processing of CH4"].loc[fixed].transpose("time").values
    for step, expected in zip(ch4, input_nodes("CH4", "RAND2020", "lowpop"), strict=True):
        assert step.parents == (expected,)


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
