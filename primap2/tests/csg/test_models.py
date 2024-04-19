import pytest

import primap2.csg
from primap2.tests.csg.utils import get_single_ts


def test_selector_match():
    da = get_single_ts(coords={"source": "A", "category": "1.A"})

    assert primap2.csg.StrategyDefinition.match(selector={"source": "A"}, fill_ts=da)
    assert not primap2.csg.StrategyDefinition.match(
        selector={"source": "B"}, fill_ts=da
    )
    assert primap2.csg.StrategyDefinition.match(
        selector={"source": "A", "category": "1.A"}, fill_ts=da
    )
    assert primap2.csg.StrategyDefinition.match(
        selector={"source": "A", "category": ["1.A", "1.B"]}, fill_ts=da
    )
    assert not primap2.csg.StrategyDefinition.match(
        selector={"source": "A", "category": "1"}, fill_ts=da
    )
    assert not primap2.csg.StrategyDefinition.match(
        selector={"source": "A", "category": ["1", "2"]}, fill_ts=da
    )


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
    da = get_single_ts(coords={"source": "A", "category": "1.A"})

    assert (
        primap2.csg.StrategyDefinition(
            [({"source": "A", "category": "1"}, 1), ({"source": "A"}, 2)]
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


def test_priority_limit():
    pd = primap2.csg.PriorityDefinition(
        priority_dimensions=["a", "b"],
        priorities=[
            {"a": "1", "b": "2", "c": "3", "d": ["4", "5"]},
            {"a": "2", "b": "3"},
        ],
    )
    assert pd.limit("e", "3") == pd
    assert pd.limit("c", "3").priorities == [
        {"a": "1", "b": "2", "d": ["4", "5"]},
        {"a": "2", "b": "3"},
    ]
    assert pd.limit("c", "4").priorities == [{"a": "2", "b": "3"}]
    assert pd.limit("d", "4").priorities == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "2", "b": "3"},
    ]
    assert pd.limit("d", "5") == pd.limit("d", "4")
    assert pd.limit("d", "6").priorities == [{"a": "2", "b": "3"}]


def test_priority_check():
    primap2.csg.PriorityDefinition(
        priority_dimensions=["a", "b"],
        priorities=[
            {"a": "1", "b": "2", "c": "3", "d": ["4", "5"]},
            {"a": "2", "b": "3"},
        ],
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
