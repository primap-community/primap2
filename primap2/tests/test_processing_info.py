"""Tests for _processing_info.py"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2
from primap2._processing_info import (
    ProcessingStepDescription,
    add_processing_step,
    combine_histories,
    record_step,
)
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


def test_processing_step_changed_dimensions():
    ds = processing_ds()

    with (
        pytest.raises(ValueError, match="Dimensions of 'CO2' changed"),
        ds.pr.processing_step(function="f", description_template="d") as s,
    ):
        s.ds = s.ds.sum("area (ISO3)")


def test_processing_step_selection():
    ds = processing_ds()

    with ds.pr.processing_step(function="f", description_template="d") as s:
        s.ds = s.ds.pr.loc[{"area": ["COL"]}]

    assert list(s.result["Processing of CO2"].values) == [step("a")]


def other_ds(*, dims=("area (ISO3)",), with_processing_info=True) -> xr.Dataset:
    """CO2 for COL and MEX, and CH4 for COL."""
    time = pd.date_range("2000", "2002", freq="YS")
    if dims:
        coords = {"area (ISO3)": ["COL", "MEX"], "time": time}
        co2 = [[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]
        co2_steps = np.array([step("c"), step("d")], dtype=object)
    else:
        coords = {"time": time}
        co2 = [5.0, 5.0, 5.0]
        co2_steps = np.array(step("c"), dtype=object)
    ds = xr.Dataset(
        {
            "CO2": xr.DataArray(
                co2,
                dims=[*dims, "time"],
                coords=coords,
                attrs={"entity": "CO2", "units": "Gg CO2 / year"},
            ),
            "CH4": xr.DataArray(
                [[1.0, 1.0, 1.0]],
                dims=["area (ISO3)", "time"],
                coords={"area (ISO3)": ["COL"], "time": time},
                attrs={"entity": "CH4", "units": "Gg CH4 / year"},
            ),
        },
        attrs={"area": "area (ISO3)"},
    )
    if with_processing_info:
        ds["Processing of CO2"] = xr.DataArray(
            co2_steps,
            dims=list(dims),
            coords={dim: coords[dim] for dim in dims},
            attrs={"entity": "Processing of CO2", "described_variable": "CO2"},
        )
        ds["Processing of CH4"] = xr.DataArray(
            np.array([step("e")], dtype=object),
            dims=["area (ISO3)"],
            coords={"area (ISO3)": ["COL"]},
            attrs={"entity": "Processing of CH4", "described_variable": "CH4"},
        )
    return ds


def test_processing_step_other_ds():
    ds = processing_ds()
    other = other_ds()

    with ds.pr.processing_step(
        function="combine", description_template="combined", other_ds=other
    ) as s:
        s.ds = s.ds.combine_first(other.pr.remove_processing_info())

    processing = s.result["Processing of CO2"]
    # changed: the step combines both histories
    assert processing.pr.loc[{"area": "COL"}].item() == ProcessingStepDescription(
        time=TimeRange("2001", "2001"),
        function="combine",
        description="combined",
        parents=(step("a"), step("c")),
    )
    # unchanged: the history of ds
    assert processing.pr.loc[{"area": "ARG"}].item() == step("b")
    # only in other_ds: the history of other_ds
    assert processing.pr.loc[{"area": "MEX"}].item() == step("d")
    # variables only in other_ds keep their history
    assert s.result["Processing of CH4"].pr.loc[{"area": "COL"}].item() == step("e")


def test_processing_step_other_ds_broadcast():
    """other_ds may lack dimensions of ds."""
    ds = processing_ds()
    other = other_ds(dims=())

    with ds.pr.processing_step(function="fill", description_template="filled", other_ds=other) as s:
        s.ds = s.ds.fillna(other[["CO2"]])

    col = s.result["Processing of CO2"].pr.loc[{"area": "COL"}].item()
    assert col.parents == (step("a"), step("c"))


def test_processing_step_other_ds_without_processing_info():
    """Data taken from another dataset needs processing information."""
    with pytest.raises(ValueError, match="the data it is combined with has none"):
        processing_ds().pr.processing_step(
            function="f",
            description_template="d",
            other_ds=other_ds(with_processing_info=False),
        )


def test_processing_step_new_timeseries():
    ds = processing_ds()

    # timeseries without data have no history
    with ds.pr.processing_step(function="f", description_template="d") as s:
        s.ds = s.ds.reindex({"area (ISO3)": ["COL", "ARG", "BOL"]})
    assert s.result["Processing of CO2"].pr.loc[{"area": "BOL"}].item() is None

    # timeseries with data are created by the processing step
    with ds.pr.processing_step(function="f", description_template="d") as s:
        s.ds = s.ds.reindex({"area (ISO3)": ["COL", "ARG", "BOL"]}, fill_value=1.0)
    bol = s.result["Processing of CO2"].pr.loc[{"area": "BOL"}].item()
    assert bol.function == "f"
    assert bol.parents == ()


def test_processing_step_new_variable():
    ds = processing_ds()

    with ds.pr.processing_step(function="f", description_template="d") as s:
        s.ds["CH4"] = s.ds["CO2"]

    col = s.result["Processing of CH4"].pr.loc[{"area": "COL"}].item()
    assert col.function == "f"
    assert col.parents == ()


def test_processing_step_incomplete_processing_info():
    """Timeseries with data need processing information."""
    ds = processing_ds()
    ds["Processing of CO2"].pr.loc[{"area": "COL"}] = None

    with pytest.raises(ValueError, match="contain data, but have no processing information"):
        ds.pr.processing_step(function="f", description_template="d")

    ds = processing_ds()
    ds["CH4"] = ds["CO2"]
    with pytest.raises(ValueError, match="has processing information, but not for 'CH4'"):
        ds.pr.processing_step(function="f", description_template="d")


def test_fillna_processing_info():
    ds = processing_ds()
    other = other_ds(dims=())[["CO2", "Processing of CO2"]]

    result = ds.pr.fillna(other)

    processing = result["Processing of CO2"]
    col = processing.pr.loc[{"area": "COL"}].item()
    assert col.function == "fillna"
    assert col.parents == (step("a"), step("c"))
    assert processing.pr.loc[{"area": "ARG"}].item() == step("b")


@pytest.mark.parametrize("function", ["fillna", "combine_first", "merge"])
def test_other_without_processing_info(function):
    """Data without processing information can't be combined with processing information."""
    ds = processing_ds()
    other = other_ds(with_processing_info=False)[["CO2"]]

    with pytest.raises(ValueError, match=r"the data it is combined with has none.*processing_step"):
        getattr(ds.pr, function)(other)
    with pytest.raises(TypeError, match=r"with_creation_step.*processing_step"):
        getattr(ds.pr, function)(other["CO2"])


def test_fillna_value_processing_info():
    """Filling with a single value creates the filled values."""
    ds = processing_ds().pr.quantify()

    result = ds.pr.fillna(0)

    col = result["Processing of CO2"].pr.loc[{"area": "COL"}].item()
    assert col == ProcessingStepDescription(
        time=TimeRange("2001", "2001"),
        function="fillna",
        description="filled missing values with 0 (CO2; area (ISO3)='COL')",
        parents=(step("a"),),
    )
    assert result["Processing of CO2"].pr.loc[{"area": "ARG"}].item() == step("b")


def test_with_creation_step():
    time = pd.date_range("2000", "2001", freq="YS")
    da = xr.DataArray(
        [[1.0, 2.0], [np.nan, np.nan]],
        dims=["area (ISO3)", "time"],
        coords={"area (ISO3)": ["COL", "ARG"], "time": time},
        name="CO2",
    )

    ds = da.pr.with_creation_step(
        function="estimate", description="estimated (<coords>)", source="expert"
    )

    xr.testing.assert_identical(ds["CO2"], da)
    assert ds["Processing of CO2"].attrs == {
        "entity": "Processing of CO2",
        "described_variable": "CO2",
    }
    assert ds["Processing of CO2"].loc["COL"].item() == ProcessingStepDescription(
        time=time.values,
        function="estimate",
        description="estimated (area (ISO3)='COL')",
        source="expert",
    )
    # timeseries without data have no processing information
    assert ds["Processing of CO2"].loc["ARG"].item() is None

    with pytest.raises(ValueError, match="needs a name"):
        da.rename(None).pr.with_creation_step(function="f", description="d")


def test_fillna_with_creation_step():
    ds = processing_ds()
    fill = other_ds(dims=(), with_processing_info=False)["CO2"]

    result = ds.pr.fillna(fill.pr.with_creation_step(function="f", description="d"))

    col = result["Processing of CO2"].pr.loc[{"area": "COL"}].item()
    assert col.function == "fillna"
    assert col.parents[0] == step("a")
    assert col.parents[1].function == "f"


def test_set_without_processing_info():
    ds = processing_ds()
    value = ds.pr.remove_processing_info().pr.loc[{"area": "COL"}]

    with pytest.raises(ValueError, match="the data it is combined with has none"):
        ds.pr.set("area", "BOL", value)


def test_combine_first_processing_info():
    ds = processing_ds()

    result = ds.pr.combine_first(other_ds())

    processing = result["Processing of CO2"]
    col = processing.pr.loc[{"area": "COL"}].item()
    assert col.function == "combine_first"
    assert col.parents == (step("a"), step("c"))
    assert processing.pr.loc[{"area": "ARG"}].item() == step("b")
    assert processing.pr.loc[{"area": "MEX"}].item() == step("d")
    assert result["Processing of CH4"].pr.loc[{"area": "COL"}].item() == step("e")


def test_merge_processing_info():
    ds = processing_ds()

    result = ds.pr.merge(other_ds(), error_on_discrepancy=False)

    processing = result["Processing of CO2"]
    col = processing.pr.loc[{"area": "COL"}].item()
    assert col.function == "merge"
    assert col.time == (TimeRange("2001", "2001"),)
    assert col.parents == (step("a"), step("c"))
    assert processing.pr.loc[{"area": "ARG"}].item() == step("b")
    assert processing.pr.loc[{"area": "MEX"}].item() == step("d")
    assert result["Processing of CH4"].pr.loc[{"area": "COL"}].item() == step("e")


def test_set_processing_info():
    ds = processing_ds()
    value = ds.pr.loc[{"area": "COL"}]

    # new timeseries take the history of the value
    result = ds.pr.set("area", "BOL", value)
    assert result["Processing of CO2"].pr.loc[{"area": "BOL"}].item() == step("a")

    # changed timeseries combine both histories
    result = ds.pr.set("area", "ARG", value, existing="overwrite")
    arg = result["Processing of CO2"].pr.loc[{"area": "ARG"}].item()
    assert arg.function == "set"
    assert arg.parents == (step("b"), step("a"))


def test_combine_histories():
    a, b, c = step("a"), step("b"), step("c")
    infos = xr.DataArray(
        np.array([[a, b], [c, None]], dtype=object),
        dims=["area", "category"],
        coords={"area": ["COL", "ARG"], "category": ["1", "2"]},
    )

    combined = combine_histories(infos, "category")

    assert combined.dims == ("area",)
    assert combined.pr.loc[{"area": "COL"}].item() == (a, b)
    # missing histories are left out
    assert combined.pr.loc[{"area": "ARG"}].item() == (c,)
    assert combine_histories(infos, ["area", "category"]).item() == (a, b, c)


def test_combine_histories_without_data():
    a, b = step("a"), step("b")
    infos = xr.DataArray(
        np.array([a, b], dtype=object), dims=["area"], coords={"area": ["COL", "ARG"]}
    )
    data = xr.DataArray(
        [[1.0, 2.0], [np.nan, np.nan]],
        dims=["area", "time"],
        coords={"area": ["COL", "ARG"], "time": pd.date_range("2000", "2001", freq="YS")},
    )

    assert combine_histories(infos, "area", data=data).item() == (a,)


def test_record_step():
    time = pd.date_range("2000", "2001", freq="YS")
    a, b = step("a"), step("b")
    old = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=["area", "time"],
        coords={"area": ["COL", "ARG"], "time": time},
        name="CO2",
    )
    old_infos = xr.DataArray(
        np.array([a, b], dtype=object), dims=["area"], coords={"area": ["COL", "ARG"]}
    )
    new = xr.DataArray(
        [[1.0, 5.0], [3.0, 4.0], [6.0, 6.0], [np.nan, np.nan]],
        dims=["area", "time"],
        coords={"area": ["COL", "ARG", "MEX", "BOL"], "time": time},
        name="CO2",
    )
    c = step("c")
    parent_values = np.empty(2, dtype=object)
    parent_values[:] = [(c,), (c,)]
    parents = xr.DataArray(parent_values, dims=["area"], coords={"area": ["COL", "MEX"]})

    result = record_step(
        new,
        previous=[(old, old_infos)],
        parents=[parents],
        function="f",
        description_template="d <coords>",
    )

    assert result.name == "Processing of CO2"
    assert result.attrs == {"entity": "Processing of CO2", "described_variable": "CO2"}
    # changed
    assert result.loc["COL"].item() == ProcessingStepDescription(
        time=TimeRange("2001", "2001"), function="f", description="d area='COL'", parents=(a, c)
    )
    # unchanged
    assert result.loc["ARG"].item() is b
    # new, with parents
    assert result.loc["MEX"].item().parents == (c,)
    np.testing.assert_array_equal(result.loc["MEX"].item().time_points(), time.values)
    # new, without data
    assert result.loc["BOL"].item() is None

    # new with data, without parents
    result = record_step(new, previous=[(old, old_infos)], function="f", description_template="d")
    assert result.loc["MEX"].item().parents == ()


def category_ds() -> xr.Dataset:
    """CO2 for categories 1.A, 1.B and 2 in COL and ARG, ARG has no data for 1.B."""
    time = pd.date_range("2000", "2001", freq="YS")
    areas = ["COL", "ARG"]
    categories = ["1.A", "1.B", "2"]
    data = np.ones((len(areas), len(categories), len(time)))
    data[1, 1, :] = np.nan
    return xr.Dataset(
        {
            "CO2": xr.DataArray(
                data,
                dims=["area (ISO3)", "category (IPCC2006)", "time"],
                coords={"area (ISO3)": areas, "category (IPCC2006)": categories, "time": time},
                attrs={"entity": "CO2", "units": "Gg CO2 / year"},
            ),
            "Processing of CO2": xr.DataArray(
                np.array([[step(f"{a} {c}") for c in categories] for a in areas], dtype=object),
                dims=["area (ISO3)", "category (IPCC2006)"],
                coords={"area (ISO3)": areas, "category (IPCC2006)": categories},
                attrs={"entity": "Processing of CO2", "described_variable": "CO2"},
            ),
        },
        attrs={"area": "area (ISO3)", "cat": "category (IPCC2006)"},
    ).pr.quantify()


def test_sum_processing_info():
    ds = category_ds()

    result = ds.pr.sum("category")

    processing = result["Processing of CO2"]
    assert processing.dims == ("area (ISO3)",)
    col = processing.pr.loc[{"area": "COL"}].item()
    assert col.function == "sum"
    assert col.parents == (step("COL 1.A"), step("COL 1.B"), step("COL 2"))
    # timeseries without data are not parents
    arg = processing.pr.loc[{"area": "ARG"}].item()
    assert arg.parents == (step("ARG 1.A"), step("ARG 2"))


@pytest.mark.parametrize("dim", ["time", "entity"])
def test_sum_processing_info_not_supported(dim):
    with pytest.raises(NotImplementedError, match="remove_processing_info"):
        category_ds().pr.sum(dim)


def test_add_aggregates_coordinates_processing_info():
    ds = category_ds()

    result = ds.pr.add_aggregates_coordinates(
        agg_info={"category": {"1": ["1.A", "1.B"], "0": ["1", "2"]}}
    )

    processing = result["Processing of CO2"]
    one = processing.pr.loc[{"area": "COL", "category": "1"}].item()
    assert one.function == "add_aggregates_coordinates"
    assert one.parents == (step("COL 1.A"), step("COL 1.B"))
    # later rules build on the results of earlier rules
    zero = processing.pr.loc[{"area": "COL", "category": "0"}].item()
    assert zero.parents == (one, step("COL 2"))
    # the sources are unchanged
    assert processing.pr.loc[{"area": "COL", "category": "1.A"}].item() == step("COL 1.A")
    # timeseries without data are not parents
    arg_one = processing.pr.loc[{"area": "ARG", "category": "1"}].item()
    assert arg_one.parents == (step("ARG 1.A"),)


def test_add_aggregates_coordinates_existing_aggregate_processing_info():
    """Existing values of the aggregate are kept, missing values are filled."""
    ds = category_ds()
    ds = ds.pr.set("category", "1", ds.pr.loc[{"category": ["1.A", "1.B"]}].pr.sum("category"))
    missing = (
        (ds["area (ISO3)"] == "COL")
        & (ds["category (IPCC2006)"] == "1")
        & (ds["time"] == np.datetime64("2001-01-01"))
    )
    ds["CO2"] = ds["CO2"].where(~missing)
    existing = ds["Processing of CO2"].pr.loc[{"area": "COL", "category": "1"}].item()

    result = ds.pr.add_aggregates_coordinates(agg_info={"category": {"1": ["1.A", "1.B"]}})

    one = result["Processing of CO2"].pr.loc[{"area": "COL", "category": "1"}].item()
    assert one.time == (TimeRange("2001", "2001"),)
    assert one.parents == (existing, step("COL 1.A"), step("COL 1.B"))


def test_add_aggregates_coordinates_several_variables_processing_info():
    """Merging the first variable adds the new coordinate to all variables."""
    ds = category_ds()
    ch4 = ds["CO2"].pint.dequantify()
    ch4.attrs.update(entity="CH4", units="Gg CH4 / year")
    ds["CH4"] = ch4.pint.quantify(unit_registry=primap2.ureg)
    ds["Processing of CH4"] = ds["Processing of CO2"].assign_attrs(
        entity="Processing of CH4", described_variable="CH4"
    )

    result = ds.pr.add_aggregates_coordinates(agg_info={"category": {"1": ["1.A", "1.B"]}})

    for var in ("CO2", "CH4"):
        one = result[f"Processing of {var}"].pr.loc[{"area": "COL", "category": "1"}].item()
        assert one.parents == (step("COL 1.A"), step("COL 1.B"))


def test_fill_padded_timeseries_processing_info(basket_processing_ds):
    """Filling a timeseries which was only added as padding records its history."""
    ds = basket_processing_ds[["CO2", "Processing of CO2", "CH4", "Processing of CH4"]]
    col = ds.pr.loc[{"area": ["COL"]}]
    mex = ds.pr.loc[{"area": ["MEX"]}]

    # CO2 for MEX pads CH4 with an empty timeseries for MEX
    padded = col.pr.combine_first(mex[["CO2", "Processing of CO2"]])
    assert padded["Processing of CH4"].pr.loc[{"area": "MEX"}].isnull()

    filled = padded.pr.fillna(mex[["CH4", "Processing of CH4"]])

    assert (
        filled["Processing of CH4"].pr.loc[{"area": "MEX"}].item()
        == mex["Processing of CH4"].pr.loc[{"area": "MEX"}].item()
    )


def test_other_with_incomplete_processing_info():
    ds = processing_ds()
    other = other_ds()
    other["Processing of CO2"].pr.loc[{"area": "COL"}] = None

    with pytest.raises(ValueError, match=r"combined with is incomplete.*processing_step"):
        ds.pr.combine_first(other)


def test_dataset_with_creation_step(basket_processing_ds):
    ds = basket_processing_ds.pr.remove_processing_info()

    result = ds.pr.with_creation_step(
        function="read", description="read <var> (<coords>)", source="RAND2020"
    )

    result.pr.ensure_valid()
    co2 = result["Processing of CO2"].pr.loc[{"area": "COL"}].item()
    assert co2 == ProcessingStepDescription(
        time=[np.datetime64("2000"), np.datetime64("2004")],
        function="read",
        description="read CO2 (area (ISO3)='COL', source='RAND2020')",
        source="RAND2020",
    )
    # the input is not modified
    assert not ds.pr.has_processing_info()

    with pytest.raises(ValueError, match="already has processing information"):
        result.pr.with_creation_step(function="read", description="read")


@primap2.records_processing_step(
    "ds", description_template="filled with {value} (<var>; <coords>)", source="{source}"
)
def fill_value(ds: xr.Dataset, value: float = 0.0, source: str = "default") -> xr.Dataset:
    assert not ds.pr.has_processing_info()
    return ds.fillna(value)


def test_records_processing_step():
    ds = processing_ds()

    result = fill_value(ds, source="test")

    np.testing.assert_array_equal(result["CO2"], [[1, 0, 3], [1, 2, 3]])
    processing = result["Processing of CO2"]
    assert processing.pr.loc[{"area": "COL"}].item() == ProcessingStepDescription(
        time=TimeRange("2001", "2001"),
        function="fill_value",
        description="filled with 0.0 (CO2; area (ISO3)='COL')",
        source="test",
        parents=(step("a"),),
    )
    assert processing.pr.loc[{"area": "ARG"}].item() == step("b")
    # the input is not modified
    assert np.isnan(ds["CO2"].pr.loc[{"area": "COL", "time": "2001"}].item())


def test_records_processing_step_without_processing_info():
    ds = processing_ds().pr.remove_processing_info()

    result = fill_value(value=1.0, ds=ds)

    assert not result.pr.has_processing_info()
    assert result["CO2"].pr.loc[{"area": "COL", "time": "2001"}].item() == 1.0


def test_records_processing_step_in_place_changes():
    @primap2.records_processing_step("ds", function="set", description_template="set")
    def set_in_place(ds: xr.Dataset) -> xr.Dataset:
        ds["CO2"].loc[{"area (ISO3)": "ARG", "time": "2000"}] = 5.0
        return ds

    ds = processing_ds()

    result = set_in_place(ds)

    assert result["Processing of CO2"].pr.loc[{"area": "ARG"}].item().function == "set"
    assert ds["CO2"].pr.loc[{"area": "ARG", "time": "2000"}].item() == 1.0


class FillAccessor(primap2._accessor_base.BaseDatasetAccessor):
    @primap2.records_processing_step(
        "self", other="other", description_template="filled (<var>; <coords>)"
    )
    def fill(self, other: xr.Dataset | xr.DataArray | float) -> xr.Dataset:
        assert isinstance(self, FillAccessor)
        assert not self._ds.pr.has_processing_info()
        if isinstance(other, xr.Dataset):
            assert not other.pr.has_processing_info()
        return self._ds.fillna(other)


def test_records_processing_step_accessor_other():
    ds = processing_ds()

    result = FillAccessor(ds).fill(other_ds(dims=())[["CO2", "Processing of CO2"]])

    col = result["Processing of CO2"].pr.loc[{"area": "COL"}].item()
    assert col.function == "FillAccessor.fill"
    assert col.parents == (step("a"), step("c"))


def test_records_processing_step_other_value():
    result = FillAccessor(processing_ds()).fill(0.0)

    col = result["Processing of CO2"].pr.loc[{"area": "COL"}].item()
    assert col.parents == (step("a"),)


def test_records_processing_step_other_invalid():
    ds = processing_ds()
    with pytest.raises(TypeError, match="DataArray it is combined with"):
        FillAccessor(ds).fill(ds["CO2"])
    with pytest.raises(ValueError, match="the data it is combined with has none"):
        FillAccessor(ds).fill(other_ds(with_processing_info=False))


def test_records_processing_step_invalid():
    with pytest.raises(ValueError, match="has no argument 'da'"):
        primap2.records_processing_step("da", description_template="d")(fill_value.__wrapped__)

    with pytest.raises(TypeError, match="has to be a Dataset, not DataArray"):
        fill_value(processing_ds()["CO2"])

    @primap2.records_processing_step("ds", description_template="d")
    def to_array(ds: xr.Dataset) -> xr.Dataset:
        return ds["CO2"]

    with pytest.raises(TypeError, match="has to return a Dataset, not DataArray"):
        to_array(processing_ds())
