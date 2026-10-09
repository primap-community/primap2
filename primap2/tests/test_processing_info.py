"""Tests for _processing_info.py"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from primap2._processing_info import ProcessingStepDescription, add_processing_step
from primap2._time_range import TimeRange


def step(name: str, *parents: ProcessingStepDescription) -> ProcessingStepDescription:
    return ProcessingStepDescription(
        time=TimeRange("2000", "2020"), function=name, description=f"step {name}", parents=parents
    )


def diamond() -> ProcessingStepDescription:
    """A history in which "a" is reachable from "d" via "b" and via "c"."""
    a = step("a")
    return step("d", step("b", a), step("c", a))


def test_parents_default_empty():
    assert step("a").parents == ()


def test_parents_converted_to_tuple():
    a = step("a")
    b = ProcessingStepDescription(time=(), function="b", description="b", parents=[a])
    assert b.parents == (a,)


def test_history_chain():
    a = step("a")
    b = step("b", a)
    c = step("c", b)
    assert c.history() == [a, b, c]


def test_history_diamond():
    d = diamond()
    # "a" is contained only once, and before both of its children
    assert [s.function for s in d.history()] == ["a", "b", "c", "d"]


def test_history_parent_order():
    a = step("a")
    c = step("c", a)
    # "c" builds on "a", so "a" has to come first even though it is the second parent
    d = step("d", c, a)
    assert [s.function for s in d.history()] == ["a", "c", "d"]


def test_history_long_chain():
    """Long histories must not exhaust the stack."""
    last = step("0")
    for i in range(1, 5000):
        last = step(str(i), last)
    assert len(last.history()) == 5000


def test_format_history():
    assert diamond().format_history() == (
        "[1] Using function=a for times=2000 to 2020: step a\n"
        "[2] (from [1]) Using function=b for times=2000 to 2020: step b\n"
        "[3] (from [1]) Using function=c for times=2000 to 2020: step c\n"
        "[4] (from [2], [3]) Using function=d for times=2000 to 2020: step d"
    )


def test_serialize_round_trip():
    d = diamond()

    result = ProcessingStepDescription.deserialize(d.serialize())

    assert result == d
    # the common ancestor is stored once and shared again after reading
    b, c = result.parents
    assert b.parents[0] is c.parents[0]


def test_serialize_round_trip_times():
    a = ProcessingStepDescription(
        time=(
            TimeRange("1990-01", "1990-12"),
            TimeRange("2000", "2020", np.timedelta64(5, "Y")),
        ),
        function="a",
        description="step a",
        source="source",
    )
    b = step("b", a)

    result = ProcessingStepDescription.deserialize(b.serialize())

    (result_a,) = result.parents
    assert result_a == a
    assert str(result_a.time[0]) == "1990-01 to 1990-12"


def test_structure_does_not_modify_input():
    u = step("a").unstructure()
    before = dict(u)

    ProcessingStepDescription.structure(u)

    assert u == before


def test_serialize_optional_missing():
    assert ProcessingStepDescription.serialize_optional(None) == b""
    assert ProcessingStepDescription.serialize_optional(np.nan) == b""
    assert ProcessingStepDescription.deserialize(b"") is None


def test_add_processing_step():
    a = step("a")
    da = xr.DataArray(np.array([a, None], dtype=object), dims=["area"])

    result = add_processing_step(da, step("b"))

    assert result.values[0] == step("b", a)
    assert result.values[1] is None
    # the input is not modified
    assert da.values[0] is a


def test_time_points_converted():
    step = ProcessingStepDescription(
        time=pd.date_range("1970", "2015", freq="YS").values, function="f", description="d"
    )
    assert step.time == (TimeRange("1970", "2015"),)


def test_time_all_rejected():
    with pytest.raises(TypeError):
        ProcessingStepDescription(time="all", function="f", description="d")
