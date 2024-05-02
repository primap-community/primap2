import pytest

import primap2.csg
from primap2 import Not
from primap2.csg._models import match_selector
from primap2.tests.csg.utils import get_single_ts


def test_match_selector():
    da = get_single_ts(coords={"source": "A", "category": "1.A"}, entity="SF6")

    assert match_selector(selector={"source": "A"}, ts=da)
    assert not match_selector(selector={"source": "B"}, ts=da)
    assert match_selector(selector={"source": "A", "category": "1.A"}, ts=da)
    assert match_selector(selector={"source": "A", "category": ["1.A", "1.B"]}, ts=da)
    assert not match_selector(selector={"source": "A", "category": "1"}, ts=da)
    assert not match_selector(selector={"source": "A", "category": ["1", "2"]}, ts=da)

    assert match_selector(selector={"source": "A", "entity": "SF6"}, ts=da)
    assert match_selector(selector={"source": "A", "entity": ["SF6", "CO2"]}, ts=da)
    assert not match_selector(selector={"source": "A", "entity": "CO2"}, ts=da)
    assert match_selector(selector={"source": "A", "variable": "SF6"}, ts=da)

    da = get_single_ts(
        coords={"source": "A", "category": "1.A"}, entity="SF6", gwp_context="AR6GWP100"
    )
    assert match_selector(selector={"source": "A", "entity": "SF6"}, ts=da)
    assert match_selector(selector={"source": "A", "entity": ["SF6", "CO2"]}, ts=da)
    assert not match_selector(selector={"source": "A", "entity": "CO2"}, ts=da)
    assert match_selector(
        selector={"source": "A", "variable": "SF6 (AR6GWP100)"}, ts=da
    )
    assert not match_selector(selector={"source": "A", "variable": "SF6"}, ts=da)


def test_selector_match_single_dim():
    assert primap2.csg.StrategyDefinition.match_single_dim(
        selector={"source": "A"}, dim="source", value="A"
    )
    assert not primap2.csg.StrategyDefinition.match_single_dim(
        selector={"source": "B"}, dim="source", value="A"
    )
    assert primap2.csg.StrategyDefinition.match_single_dim(
        selector={"source": ["A", "B"], "category": "1.A"}, dim="source", value="A"
    )
    assert primap2.csg.StrategyDefinition.match_single_dim(
        selector={"source": "A", "category": "1"}, dim="other", value="any"
    )


def test_strategy_definition():
    da = get_single_ts(
        coords={"source": "A", "category": "1.A"}, entity="SF6", gwp_context="AR6GWP100"
    )

    assert (
        primap2.csg.StrategyDefinition(
            [({"source": "A", "category": "1"}, 1), ({"source": "A"}, 2)]
        ).find_strategy(da)
        == 2
    )
    assert (
        primap2.csg.StrategyDefinition(
            [
                ({"source": "A", "category": "1.A", "entity": ["SF6", "CO2"]}, 1),
                ({"source": "A"}, 2),
            ]
        ).find_strategy(da)
        == 1
    )
    assert (
        primap2.csg.StrategyDefinition(
            [
                ({"source": "A", "category": "1.A", "entity": ["CH4", "CO2"]}, 1),
                ({"source": "A"}, 2),
            ]
        ).find_strategy(da)
        == 2
    )
    assert (
        primap2.csg.StrategyDefinition(
            [
                (
                    {
                        "source": "A",
                        "category": "1.A",
                        "variable": ["SF6 (AR6GWP100)", "CO2"],
                    },
                    1,
                ),
                ({"source": "A"}, 2),
            ]
        ).find_strategy(da)
        == 1
    )
    assert (
        primap2.csg.StrategyDefinition(
            [
                (
                    {
                        "source": "A",
                        "category": "1.A",
                        "variable": ["SF6 (AR4GWP100)", "CO2"],
                    },
                    1,
                ),
                ({"source": "A"}, 2),
            ]
        ).find_strategy(da)
        == 2
    )
    assert (
        primap2.csg.StrategyDefinition(
            [
                ({"source": "A", "category": "1"}, 1),
                ({"source": "A", "category": "1.A"}, 2),
            ]
        ).find_strategy(da)
        == 2
    )
    with pytest.raises(KeyError):
        primap2.csg.StrategyDefinition(
            [
                ({"source": "A", "category": "1"}, 1),
                ({"source": "A", "category": "1.B"}, 2),
                ({"source": "B", "category": "1.B"}, 3),
            ]
        ).find_strategy(da)


def test_strategy_definition_limit():
    assert primap2.csg.StrategyDefinition(
        [({"entity": "A", "source": "S"}, 1), ({"source": "T"}, 2)]
    ).limit("entity", "A").strategies == [({"source": "S"}, 1), ({"source": "T"}, 2)]
    assert primap2.csg.StrategyDefinition(
        [({"entity": "A", "source": "S"}, 1), ({"source": "T"}, 2)]
    ).limit("entity", "B").strategies == [({"source": "T"}, 2)]


def test_strategy_definition_check_dimensions(minimal_ds):
    primap2.csg.StrategyDefinition(
        [({"entity": "CO2", "source": "RAND2020"}, 1)]
    ).check_dimensions(minimal_ds)
    primap2.csg.StrategyDefinition(
        [({"variable": "CO2", "source": "RAND2020"}, 1)]
    ).check_dimensions(minimal_ds)
    with pytest.raises(ValueError):
        primap2.csg.StrategyDefinition(
            [({"entity": "CO2", "scenario": "highpop"})]
        ).check_dimensions(minimal_ds)


def test_priority_limit():
    pd = primap2.csg.PriorityDefinition(
        priority_dimensions=["a", "b"],
        priorities=[
            {
                "a": "1",
                "b": "2",
                "c": "3",
                "d": ["4", "5"],
                "e": Not("6"),
                "f": Not(["7", "8"]),
            },
            {"a": "2", "b": "3"},
        ],
        exclude_result=[{"c": "4"}],
        exclude_input=[{"a": "1", "c": "3", "d": "4"}],
    )
    assert pd.limit("g", "3") == pd
    assert pd.limit("c", "3").priorities == [
        {"a": "1", "b": "2", "d": ["4", "5"], "e": Not("6"), "f": Not(["7", "8"])},
        {"a": "2", "b": "3"},
    ]
    assert pd.limit("c", "4").priorities == [{"a": "2", "b": "3"}]
    assert pd.limit("d", "4").priorities == [
        {"a": "1", "b": "2", "c": "3", "e": Not("6"), "f": Not(["7", "8"])},
        {"a": "2", "b": "3"},
    ]
    assert pd.limit("d", "5") == pd.limit("d", "4")
    assert pd.limit("d", "6").priorities == [{"a": "2", "b": "3"}]
    assert pd.limit("e", "7").priorities == [
        {"a": "1", "b": "2", "c": "3", "d": ["4", "5"], "f": Not(["7", "8"])},
        {"a": "2", "b": "3"},
    ]
    assert pd.limit("e", "6").priorities == [{"a": "2", "b": "3"}]
    assert pd.limit("f", "6").priorities == [
        {"a": "1", "b": "2", "c": "3", "d": ["4", "5"], "e": Not("6")},
        {"a": "2", "b": "3"},
    ]
    assert pd.limit("f", "7").priorities == [{"a": "2", "b": "3"}]
    assert pd.limit("f", "7") == pd.limit("f", "8")


def test_priority_exclude_result():
    pd = primap2.csg.PriorityDefinition(
        priority_dimensions=["a"],
        priorities=[
            {"a": "1"},
            {"a": "2"},
        ],
        exclude_result=[{"b": "3", "c": ["4", "5"]}, {"c": "6"}],
    )
    pd.check_dimensions()

    assert pd.excludes_result(get_single_ts(coords={"a": "1", "b": "3", "c": "4"}))
    assert pd.excludes_result(get_single_ts(coords={"a": "1", "b": "3", "c": "5"}))
    assert pd.excludes_result(get_single_ts(coords={"a": "1", "b": "3", "c": "6"}))
    assert pd.excludes_result(get_single_ts(coords={"a": "1", "b": "4", "c": "6"}))
    assert not pd.excludes_result(get_single_ts(coords={"a": "1", "b": "3", "c": "7"}))
    assert not pd.excludes_result(get_single_ts(coords={"a": "1", "b": "4", "c": "4"}))


def test_priority_exclude_input():
    pd = primap2.csg.PriorityDefinition(
        priority_dimensions=["a"],
        priorities=[
            {"a": "1"},
            {"a": "2"},
        ],
        exclude_input=[
            {"a": "1", "b": "3", "c": ["4", "5"]},
            {"c": "6"},
        ],
    )
    pd.check_dimensions()

    assert pd.excludes_input(get_single_ts(coords={"a": "1", "b": "3", "c": "4"}))
    assert pd.excludes_input(get_single_ts(coords={"a": "1", "b": "3", "c": "5"}))
    assert pd.excludes_input(get_single_ts(coords={"a": "1", "b": "3", "c": "6"}))
    assert pd.excludes_input(get_single_ts(coords={"a": "1", "b": "4", "c": "6"}))
    assert not pd.excludes_input(get_single_ts(coords={"a": "2", "b": "3", "c": "4"}))
    assert not pd.excludes_input(get_single_ts(coords={"a": "1", "b": "3", "c": "7"}))
    assert not pd.excludes_input(get_single_ts(coords={"a": "1", "b": "4", "c": "4"}))


def test_priority_check():
    primap2.csg.PriorityDefinition(
        priority_dimensions=["a", "b"],
        priorities=[
            {"a": "1", "b": "2", "c": "3", "d": ["4", "5"]},
            {"a": "2", "b": "3"},
        ],
        exclude_result=[{"c": "5"}, {"d": ["6", "7"]}],
    ).check_dimensions()

    with pytest.raises(ValueError):
        primap2.csg.PriorityDefinition(
            priority_dimensions=["a", "b"],
            priorities=[
                {"a": "1", "b": "2", "c": "3", "d": ["4", "5"]},
                {"a": "2"},
            ],
        ).check_dimensions()

    with pytest.raises(ValueError):
        primap2.csg.PriorityDefinition(
            priority_dimensions=["a", "b"],
            priorities=[
                {"a": "1", "b": "2", "c": "3", "d": ["4", "5"]},
                {"a": "2", "b": ["2", "3"]},
            ],
        ).check_dimensions()

    with pytest.raises(ValueError):
        primap2.csg.PriorityDefinition(
            priority_dimensions=["a", "b"],
            priorities=[
                {"a": "1", "b": "2", "c": "3", "d": ["4", "5"]},
                {"a": "2", "b": "3"},
            ],
            exclude_result=[{"c": "5"}, {"d": ["6", "7"], "a": "3"}],
        ).check_dimensions()
