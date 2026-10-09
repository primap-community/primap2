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
    assert result.values[0].parents[0] is a
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


def processing_ds() -> xr.Dataset:
    time = pd.date_range("2000", "2002", freq="YS")
    return xr.Dataset(
        {
            "CO2": xr.DataArray(
                [[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]],
                dims=["area (ISO3)", "time"],
                coords={"area (ISO3)": ["COL", "ARG"], "time": time},
                attrs={"entity": "CO2", "units": "Gg CO2 / year"},
            ),
            "Processing of CO2": xr.DataArray(
                np.array([step("a"), step("b")], dtype=object),
                dims=["area (ISO3)"],
                coords={"area (ISO3)": ["COL", "ARG"]},
                attrs={"entity": "Processing of CO2", "described_variable": "CO2"},
            ),
        },
        attrs={"area": "area (ISO3)"},
    )


def test_processing_step_records_changes():
    ds = processing_ds()

    with ds.pr.processing_step(function="fill", description_template="filled <coords>") as s:
        assert not s.ds.pr.has_processing_info()
        s.ds = s.ds.fillna(0)

    result = s.result
    np.testing.assert_array_equal(result["CO2"], [[1, 0, 3], [1, 2, 3]])
    col = result["Processing of CO2"].pr.loc[{"area": "COL"}].item()
    assert col == ProcessingStepDescription(
        time=TimeRange("2001", "2001"),
        function="fill",
        description="filled area (ISO3)='COL'",
        parents=(step("a"),),
    )
    # ARG was not changed
    assert result["Processing of CO2"].pr.loc[{"area": "ARG"}].item() == step("b")


def test_processing_step_shares_history():
    """Steps are immutable, so the history must not be copied."""
    ds = processing_ds()
    col_before, arg_before = ds["Processing of CO2"].values

    with ds.pr.processing_step(function="fill", description_template="filled") as s:
        s.ds = s.ds.fillna(0)

    col, arg = s.result["Processing of CO2"].values
    assert col.parents[0] is col_before
    assert arg is arg_before
    # the input is not modified
    assert ds["Processing of CO2"].values[0] is col_before


def test_processing_step_in_place_changes():
    ds = processing_ds()

    with ds.pr.processing_step(function="set", description_template="set") as s:
        s.ds["CO2"].loc[{"area (ISO3)": "ARG", "time": "2000"}] = 5.0

    assert s.result["Processing of CO2"].pr.loc[{"area": "ARG"}].item().function == "set"
    # the input is not modified
    assert ds["CO2"].pr.loc[{"area": "ARG", "time": "2000"}].item() == 1.0


def test_processing_step_no_change_logged(caplog):
    ds = processing_ds()

    with ds.pr.processing_step(function="nothing", description_template="nothing") as s:
        pass

    xr.testing.assert_identical(s.result, ds)
    assert "No data changed in the processing step of 'nothing'" in caplog.text


def test_processing_step_exception():
    ds = processing_ds()

    with (
        pytest.raises(KeyError),
        ds.pr.processing_step(function="f", description_template="d") as s,
    ):
        raise KeyError("error")

    with pytest.raises(RuntimeError, match="only available after the with block"):
        _ = s.result


def test_processing_step_result_within_block():
    ds = processing_ds()

    with (
        ds.pr.processing_step(function="f", description_template="d") as s,
        pytest.raises(RuntimeError),
    ):
        _ = s.result


def test_processing_step_changed_coordinates():
    ds = processing_ds()

    with (
        pytest.raises(ValueError, match="Coordinate values"),
        ds.pr.processing_step(function="f", description_template="d") as s,
    ):
        s.ds = s.ds.pr.loc[{"area": ["COL"]}]
