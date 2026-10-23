"""Tests for _processing_info.py"""

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2
from primap2._processing_info import (
    ProcessingStep,
    ProcessingStepDescription,
    _format_time,
    _label,
    add_processing_step,
    combine_histories,
    ensure_complete_processing_info,
    record_step,
    used_sources,
)


def desc(name: str, source: str | None = None) -> ProcessingStepDescription:
    return ProcessingStepDescription(function=name, description=f"step {name}", source=source)


def step(name: str, *parents: ProcessingStep, time: str = "2000") -> ProcessingStep:
    return ProcessingStep(desc(name), np.datetime64(time), parents)


def diamond() -> ProcessingStep:
    """A history in which "a" is reachable from "d" via "b" and via "c"."""
    a = step("a")
    return step("d", step("b", a), step("c", a))


def test_description_ids_are_unique():
    assert desc("a").id != desc("a").id
    assert desc("a") != desc("a")


def test_description_coords_copied():
    coords = {"area": "COL"}
    d = ProcessingStepDescription(function="f", description="d", coords=coords)
    coords["area"] = "ARG"
    assert d.coords == {"area": "COL"}


def test_parents_default_empty():
    assert step("a").parents == ()


def test_parents_converted_to_tuple():
    a = step("a")
    b = ProcessingStep(desc("b"), np.datetime64("2000"), [a])
    assert b.parents == (a,)


def test_equality_ignores_parents():
    d = desc("a")
    time = np.datetime64("2000")
    first = ProcessingStep(d, time, (step("x"),))
    second = ProcessingStep(d, time, ())
    assert first == second
    assert hash(first) == hash(second)
    assert first != ProcessingStep(d, np.datetime64("2001"))
    assert first != ProcessingStep(desc("a"), time)


def test_no_graph_walking():
    """Comparing, hashing and printing deep histories is cheap and doesn't recurse."""

    def chain() -> ProcessingStep:
        d = desc("x")
        last = ProcessingStep(d, np.datetime64("2000"))
        for _ in range(50_000):
            last = ProcessingStep(d, np.datetime64("2000"), (last,))
        return last

    first = chain()
    assert first == first  # noqa: PLR0124
    assert first != chain()
    assert isinstance({first}, set)
    assert "parents" not in repr(first)


def test_history_chain():
    a = step("a")
    b = step("b", a)
    c = step("c", b)
    assert c.history() == [a, b, c]
    assert all(x is y for x, y in zip(c.history(), [a, b, c], strict=True))


def test_history_diamond():
    d = diamond()
    # "a" is contained only once, and before both of its children
    assert [s.description.function for s in d.history()] == ["a", "b", "c", "d"]


def test_history_parent_order():
    a = step("a")
    c = step("c", a)
    # "c" builds on "a", so "a" has to come first even though it is the second parent
    d = step("d", c, a)
    assert [s.description.function for s in d.history()] == ["a", "c", "d"]


def test_history_long_chain():
    """Long histories must not exhaust the stack."""
    last = step("0")
    for i in range(1, 5000):
        last = step(str(i), last)
    assert len(last.history()) == 5000


def test_history_links_to_other_times():
    early = [step("a", time=str(year)) for year in (1990, 1991)]
    late = step("b", *early, time="2005")
    assert late.history() == [*early, late]


def test_format_history():
    assert diamond().format_history() == (
        "d for 2000: step d\n"
        "├─ b for 2000: step b\n"
        "│  ↑ (a) a for 2000: step a\n"
        "└─ c for 2000: step c\n"
        "   ↑ see (a)"
    )


def test_format_history_chain_and_repeated_ancestor():
    x = step("x", step("a"))
    y = step("y", step("b", x), step("c", x))
    assert step("z", y).format_history() == (
        "z for 2000: step z\n"
        "↑ y for 2000: step y\n"
        "  ├─ b for 2000: step b\n"
        "  │  ↑ (a) x for 2000: step x\n"
        "  │  ↑ a for 2000: step a\n"
        "  └─ c for 2000: step c\n"
        "     ↑ see (a)"
    )


def test_format_history_split_within_branch():
    shared = step("s1")
    m = step("m", shared, step("s2"))
    r = step("r", step("p", m), step("q", shared))
    assert r.format_history() == (
        "r for 2000: step r\n"
        "├─ p for 2000: step p\n"
        "│  ↑ m for 2000: step m\n"
        "│    ├─ (a) s1 for 2000: step s1\n"
        "│    └─ s2 for 2000: step s2\n"
        "└─ q for 2000: step q\n"
        "   ↑ see (a)"
    )


def test_format_history_other_time_points():
    parents = [step("a", time=str(year)) for year in (1990, 1991)]
    assert step("b", *parents, time="2005").format_history() == (
        "b for 2005: step b\n├─ a for 1990: step a\n└─ a for 1991: step a"
    )


@pytest.mark.parametrize(
    ("index", "label"),
    [
        (0, "a"),
        (1, "b"),
        (25, "z"),
        (26, "aa"),
        (27, "ab"),
        (51, "az"),
        (52, "ba"),
        (701, "zz"),
        (702, "aaa"),
        (703, "aab"),
        (18_277, "zzz"),
        (18_278, "aaaa"),
    ],
)
def test_label(index, label):
    assert _label(index) == label


def test_label_unique_and_ordered():
    labels = [_label(i) for i in range(20_000)]
    assert len(set(labels)) == len(labels)
    assert labels == sorted(labels, key=lambda x: (len(x), x))


def test_format_history_deep_branches():
    s = step("leaf")
    for i in range(5000):
        s = step(str(i), s, step(f"other {i}"))
    assert len(s.format_history().splitlines()) == 10001


@pytest.mark.parametrize(
    ("time", "expected"),
    [
        ("2000", "2000"),
        ("2000-01-01T00:00:00", "2000"),
        ("2000-01-01", "2000"),
        ("2000-02-01", "2000-02-01"),
        ("2000-01-02", "2000-01-02"),
        ("2000-01-01T06:00", "2000-01-01T06:00"),
    ],
)
def test_format_time(time, expected):
    assert _format_time(np.datetime64(time)) == expected


def test_str_timeseries_and_source():
    s = ProcessingStep(
        ProcessingStepDescription(
            function="estimate",
            description="expert estimate",
            source="expert",
            variable="CO2",
            coords={"area (ISO3)": "COL", "source": "RAND2020"},
        ),
        np.datetime64("2002"),
    )
    assert str(s) == (
        "CO2 area (ISO3)='COL', source='RAND2020': estimate (data source: expert) for "
        "2002: expert estimate"
    )


TIME = pd.date_range("2000", "2002", freq="YS")


def leaves(
    da: xr.DataArray, names: dict, area_dim: str = "area (ISO3)", sources: bool = False
) -> xr.DataArray:
    """A node per data point with data; the description is named by the area, which is
    also the source if ``sources``."""
    values = np.full(da.shape, None, dtype=object)
    descriptions: dict[str, ProcessingStepDescription] = {}
    for index in np.ndindex(da.shape):
        if np.isnan(da.values[index]):
            continue
        coords = {dim: da[dim].values[i] for dim, i in zip(da.dims, index, strict=True)}
        name = names[coords.get(area_dim)]
        values[index] = ProcessingStep(
            descriptions.setdefault(name, desc(name, name if sources else None)), coords["time"]
        )
    processing = f"Processing of {da.name}"
    return xr.DataArray(
        values,
        dims=da.dims,
        coords=da.coords,
        name=processing,
        attrs={"entity": processing, "described_variable": da.name},
    )


def processing_ds(sources: bool = False) -> xr.Dataset:
    """CO2 in COL (no data in 2001) and ARG, each area created by its own step."""
    ds = xr.Dataset(
        {
            "CO2": xr.DataArray(
                [[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]],
                dims=["area (ISO3)", "time"],
                coords={"area (ISO3)": ["COL", "ARG"], "time": TIME},
                attrs={"entity": "CO2", "units": "Gg CO2 / year"},
            ),
        },
        attrs={"area": "area (ISO3)"},
    )
    ds["Processing of CO2"] = leaves(ds["CO2"], {"COL": "a", "ARG": "b"}, sources=sources)
    return ds


def node_at(ds: xr.Dataset, area: str, year: str, var: str = "CO2") -> ProcessingStep | None:
    return ds[f"Processing of {var}"].pr.loc[{"area": area, "time": year}].item()


def test_record_step_creation():
    da = xr.DataArray(
        [[1.0, 2.0], [np.nan, np.nan]],
        dims=["area", "time"],
        coords={"area": ["COL", "ARG"], "time": TIME[:2]},
        name="CO2",
    )

    result = record_step(da, function="estimate", description="estimated", source="expert")

    assert result.name == "Processing of CO2"
    assert result.dims == da.dims
    assert result.attrs == {"entity": "Processing of CO2", "described_variable": "CO2"}
    first, second = result.loc["COL"].values
    assert first.time == np.datetime64("2000")
    assert second.time == np.datetime64("2001")
    assert first.parents == ()
    # one description per timeseries, shared by all its nodes
    assert first.description is second.description
    assert first.description.function == "estimate"
    assert first.description.source == "expert"
    assert first.description.variable == "CO2"
    assert first.description.coords == {"area": "COL"}
    # no data, no node
    assert (result.loc["ARG"].values == None).all()  # noqa: E711


def test_record_step_dimension_order_kept():
    da = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=["time", "area"],
        coords={"area": ["COL", "ARG"], "time": TIME[:2]},
        name="CO2",
    )

    result = record_step(da, function="f", description="d", source="s")

    assert result.dims == ("time", "area")
    assert result.loc["2001", "ARG"].item().time == np.datetime64("2001")
    assert result.loc["2001", "ARG"].item().description.coords == {"area": "ARG"}


def test_record_step_requires_time():
    da = xr.DataArray([1.0], dims=["area"], coords={"area": ["COL"]}, name="CO2")
    with pytest.raises(ValueError, match="no 'time' dimension"):
        record_step(da, function="f", description="d")


def previous_state():
    """Old data of COL and ARG for 2000 and 2001 with nodes."""
    old = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=["area", "time"],
        coords={"area": ["COL", "ARG"], "time": TIME[:2]},
        name="CO2",
    )
    return old, leaves(old, {"COL": "a", "ARG": "b"}, area_dim="area")


def test_record_step_unchanged_and_changed():
    old, old_infos = previous_state()
    new = old.copy()
    new.loc[{"area": "COL", "time": TIME[1]}] = 5.0

    result = record_step(
        new, previous=[(old, old_infos)], function="f", description="d", source="s"
    )

    # unchanged data points keep the old node object
    for area, year in [("COL", 0), ("ARG", 0), ("ARG", 1)]:
        assert result.loc[area].values[year] is old_infos.loc[area].values[year]
    changed = result.loc["COL"].values[1]
    old_node = old_infos.loc["COL"].values[1]
    assert changed.parents == (old_node,)
    assert changed.parents[0] is old_node
    assert changed.time == np.datetime64("2001")
    assert changed.description.function == "f"
    assert changed.description.coords == {"area": "COL"}


def test_record_step_replaces():
    old, old_infos = previous_state()
    new = old * 2

    kept = record_step(new, previous=[(old, old_infos)], function="f", description="d")
    replaced = record_step(
        new, previous=[(old, old_infos)], replaces=True, function="f", description="d", source="s"
    )

    assert kept.loc["COL"].values[0].parents == (old_infos.loc["COL"].values[0],)
    assert replaced.loc["COL"].values[0].parents == ()


def test_record_step_nan_removes_node():
    old, old_infos = previous_state()
    new = old.copy()
    new.loc[{"area": "COL", "time": TIME[0]}] = np.nan

    result = record_step(new, previous=[(old, old_infos)], function="f", description="d")

    assert result.loc["COL"].values[0] is None
    assert result.loc["COL"].values[1] is old_infos.loc["COL"].values[1]
    ensure_complete_processing_info(xr.Dataset({"CO2": new, "Processing of CO2": result}))


def test_record_step_new_timeseries_and_padding():
    old, old_infos = previous_state()
    areas = ["COL", "ARG", "MEX", "BOL"]
    new = old.reindex(area=areas)
    new.loc[{"area": "MEX"}] = 7.0

    result = record_step(
        new, previous=[(old, old_infos)], function="f", description="d", source="s"
    )

    mex = result.loc["MEX"].values
    assert [x.parents for x in mex] == [(), ()]
    assert mex[0].description is mex[1].description
    assert (result.loc["BOL"].values == None).all()  # noqa: E711
    assert result.loc["COL"].values[0] is old_infos.loc["COL"].values[0]


def test_record_step_first_previous_wins():
    old, old_infos = previous_state()
    other = old.copy()
    other_infos = leaves(other, {"COL": "x", "ARG": "y"}, area_dim="area")
    new = old.copy()
    new.loc[{"area": "COL", "time": TIME[0]}] = 9.0
    new = new.reindex(area=["COL", "ARG", "MEX"])
    new.loc[{"area": "MEX"}] = 1.0
    both = [(old.sel(area=["COL"]), old_infos.sel(area=["COL"])), (other, other_infos)]

    result = record_step(new, previous=both, function="f", description="d", source="s")

    # COL is found in the first entry, ARG only in the second
    assert result.loc["COL"].values[0].parents == (old_infos.loc["COL"].values[0],)
    assert result.loc["ARG"].values[0] is other_infos.loc["ARG"].values[0]


def test_record_step_mismatching_processing_info():
    old, old_infos = previous_state()
    with pytest.raises(ValueError, match="doesn't match the data it describes"):
        record_step(old, previous=[(old, old_infos.isel(time=[0]))], function="f", description="d")


def test_record_step_parents_broadcast():
    old, old_infos = previous_state()
    new = old * 2
    c = step("c")
    # lacks the time dimension and the area ARG
    by_area = xr.DataArray(np.array([c], dtype=object), dims=["area"], coords={"area": ["COL"]})
    # lacks the area dimension
    by_time = xr.DataArray(
        np.array([step("t0"), None], dtype=object), dims=["time"], coords={"time": TIME[:2]}
    )
    # lacks everything
    scalar = xr.DataArray(step("s"))

    result = record_step(
        new,
        previous=[(old, old_infos)],
        parents=[by_area, by_time, scalar],
        function="f",
        description="d",
    )

    names = lambda node: [p.description.function for p in node.parents]  # noqa: E731
    assert names(result.loc["COL"].values[0]) == ["a", "c", "t0", "s"]
    assert names(result.loc["COL"].values[1]) == ["a", "c", "s"]
    assert names(result.loc["ARG"].values[0]) == ["b", "t0", "s"]
    assert names(result.loc["ARG"].values[1]) == ["b", "s"]


def test_record_step_tuple_parents_for_other_time_points():
    time = pd.date_range("1990", "2005", freq="YS")
    new = xr.DataArray([np.nan] * 15 + [1.0], dims=["time"], coords={"time": time}, name="CO2")
    early = tuple(step("fit", time=str(year)) for year in (1990, 1995, 2000))
    cells = np.empty(len(time), dtype=object)
    cells[:] = None
    cells[-1] = early
    parents = xr.DataArray(cells, dims=["time"], coords={"time": time})

    result = record_step(new, parents=[parents], function="extrapolate", description="d")

    node = result.values[-1]
    assert node.time == np.datetime64("2005")
    assert node.parents == early
    assert all(a is b for a, b in zip(node.parents, early, strict=True))
    assert [n.time for n in node.history()[:-1]] == [e.time for e in early]


def test_record_step_parents_deduplicated():
    old, old_infos = previous_state()
    shared = old_infos.loc["COL"].values[0]
    cells = np.empty(1, dtype=object)
    cells[0] = (shared, step("n"), shared)
    parents = xr.DataArray(cells, dims=["area"], coords={"area": ["COL"]})

    result = record_step(
        old * 2, previous=[(old, old_infos)], parents=[parents], function="f", description="d"
    )

    node = result.loc["COL"].values[0]
    assert len(node.parents) == 2
    assert node.parents[0] is shared
    assert node.parents[1].description.function == "n"


def test_record_step_nan_parents_ignored():
    old, _ = previous_state()
    parents = xr.DataArray(
        np.array([np.nan, step("p")], dtype=object), dims=["area"], coords={"area": ["COL", "ARG"]}
    )

    result = record_step(old, parents=[parents], function="f", description="d", source="s")

    assert result.loc["COL"].values[0].parents == ()
    assert result.loc["ARG"].values[0].parents[0].description.function == "p"


def test_record_step_does_not_modify_inputs():
    old, old_infos = previous_state()
    before = old_infos.values.copy()
    record_step(old * 2, previous=[(old, old_infos)], function="f", description="d")
    assert all(a is b for a, b in zip(old_infos.values.flat, before.flat, strict=True))


def test_add_processing_step():
    infos = processing_ds()["Processing of CO2"]

    result = add_processing_step(infos, desc("conv"), variable="CH4")

    assert result.dims == infos.dims
    col = result.loc["COL"].values
    assert col[1] is None
    assert col[0].parents[0] is infos.loc["COL"].values[0]
    assert col[0].time == np.datetime64("2000")
    assert col[0].description.function == "conv"
    assert col[0].description.variable == "CH4"
    assert col[0].description.coords == {"area (ISO3)": "COL"}
    assert col[0].description is col[2].description
    assert col[0].description != result.loc["ARG"].values[0].description
    # the input is not modified
    assert infos.loc["COL"].values[0].description.function == "a"


def test_combine_histories():
    time = TIME[:2]
    a, b, c = step("a"), step("b"), step("c")
    cells = np.array([[[a, a], [b, None]], [[c, c], [None, None]]], dtype=object)
    infos = xr.DataArray(
        cells,
        dims=["area", "category", "time"],
        coords={"area": ["COL", "ARG"], "category": ["1", "2"], "time": time},
    )

    combined = combine_histories(infos, "category")

    assert combined.dims == ("area", "time")
    assert combined.loc["COL"].values[0] == (a, b)
    # missing histories are left out
    assert combined.loc["COL"].values[1] == (a,)
    assert combined.loc["ARG"].values[0] == (c,)
    assert combined.loc["ARG"].values[1] == (c,)
    both = combine_histories(infos, ["area", "category"])
    assert both.dims == ("time",)
    assert both.values[0] == (a, b, c)
    assert both.values[1] == (a, c)


def test_combine_histories_no_nodes():
    infos = xr.DataArray(
        np.full((2, 1), None, dtype=object),
        dims=["area", "time"],
        coords={"area": ["COL", "ARG"], "time": TIME[:1]},
    )
    assert combine_histories(infos, "area").values[0] == ()


def test_combine_histories_without_data():
    a, b = step("a"), step("b")
    infos = xr.DataArray(
        np.array([[a], [b]], dtype=object),
        dims=["area", "time"],
        coords={"area": ["COL", "ARG"], "time": TIME[:1]},
    )
    data = xr.DataArray(
        [[1.0], [np.nan]],
        dims=["area", "time"],
        coords={"area": ["COL", "ARG"], "time": TIME[:1]},
    )

    assert combine_histories(infos, "area", data=data).values[0] == (a,)


def test_with_creation_step():
    da = xr.DataArray(
        [[1.0, 2.0], [np.nan, 3.0]],
        dims=["area (ISO3)", "time"],
        coords={"area (ISO3)": ["COL", "ARG"], "time": TIME[:2]},
        name="CO2",
    )

    ds = da.pr.with_creation_step(function="estimate", description="estimated", source="expert")

    xr.testing.assert_identical(ds["CO2"], da)
    assert ds["Processing of CO2"].attrs == {
        "entity": "Processing of CO2",
        "described_variable": "CO2",
    }
    first, second = ds["Processing of CO2"].loc["COL"].values
    assert first.parents == ()
    assert first.time == np.datetime64("2000")
    assert first.description is second.description
    assert first.description.source == "expert"
    assert first.description.variable == "CO2"
    assert first.description.coords == {"area (ISO3)": "COL"}
    # one node per data point with data
    assert ds["Processing of CO2"].loc["ARG"].values[0] is None
    assert ds["Processing of CO2"].loc["ARG"].values[1].time == np.datetime64("2001")

    with pytest.raises(ValueError, match="needs a name"):
        da.rename(None).pr.with_creation_step(function="f", description="d")


def test_dataset_with_creation_step(basket_processing_ds):
    ds = basket_processing_ds.pr.without_processing_info()

    result = ds.pr.with_creation_step(function="read", description="read", source="RAND2020")

    result.pr.ensure_valid()
    co2 = result["Processing of CO2"].pr.loc[{"area": "COL"}].values.squeeze()
    # data in 2000 and 2004 only
    assert [x is not None for x in co2] == [True, False, False, False, True]
    assert co2[0].description == ProcessingStepDescription(
        id=co2[0].description.id,
        function="read",
        description="read",
        source="RAND2020",
        variable="CO2",
        coords={"area (ISO3)": "COL", "source": "RAND2020"},
    )
    assert co2[0].description is co2[4].description
    assert co2[4].time == np.datetime64("2004")
    # the input is not modified
    assert not ds.pr.has_processing_info()

    with pytest.raises(ValueError, match="already has processing information"):
        result.pr.with_creation_step(function="read", description="read")


def test_ensure_complete_no_processing_info(opulent_ds):
    ensure_complete_processing_info(opulent_ds)


def test_ensure_complete_valid(opulent_processing_ds):
    ensure_complete_processing_info(opulent_processing_ds)


def test_ensure_complete_missing_variable(opulent_processing_ds):
    ds = opulent_processing_ds.drop_vars("Processing of CO2")
    with pytest.raises(ValueError, match="has processing information, but not for 'CO2'"):
        ensure_complete_processing_info(ds)


def test_ensure_complete_wrong_dimensions(opulent_processing_ds):
    ds = opulent_processing_ds
    ds["Processing of CO2"] = ds["Processing of CO2"].isel(time=0, drop=True)
    with pytest.raises(ValueError, match="has the dimensions"):
        ensure_complete_processing_info(ds)


def test_ensure_complete_missing_node():
    ds = processing_ds()
    ds["Processing of CO2"].values[1, 2] = None
    with pytest.raises(
        ValueError, match=r"1 data points of 'CO2' contain data, but have no processing"
    ) as err:
        ensure_complete_processing_info(ds)
    assert "area (ISO3)='ARG'" in str(err.value)
    assert "time=" in str(err.value)


def test_ensure_complete_node_without_data():
    ds = processing_ds()
    ds["Processing of CO2"].values[0, 1] = step("x", time="2001")
    with pytest.raises(ValueError, match=r"1 data points of 'CO2' have processing information"):
        ensure_complete_processing_info(ds)


def test_ensure_complete_wrong_time():
    ds = processing_ds()
    ds["Processing of CO2"].values[0, 0] = step("x", time="2001")
    with pytest.raises(ValueError, match=r"1 processing steps of 'CO2' have a different time"):
        ensure_complete_processing_info(ds)


def test_ensure_complete_not_a_step():
    ds = processing_ds()
    ds["Processing of CO2"].values[0, 0] = "text"
    with pytest.raises(ValueError, match="something other than processing steps"):
        ensure_complete_processing_info(ds)


def test_ensure_valid_checks_processing_info(opulent_processing_ds):
    opulent_processing_ds["Processing of CO2"].values.flat[0] = None
    with pytest.raises(ValueError, match="contain data, but have no processing"):
        opulent_processing_ds.pr.ensure_valid()


# --- netCDF round trips


def modified_ds(ds: xr.Dataset) -> xr.Dataset:
    """Double CO2 and let SF6 depend on the new CO2, so that nodes are shared between
    variables."""
    ds = ds.copy()
    previous = [(ds["CO2"], ds["Processing of CO2"])]
    ds["CO2"] = ds["CO2"] * 2
    ds["Processing of CO2"] = record_step(
        ds["CO2"], previous=previous, function="double", description="doubled"
    )
    ds["SF6"] = ds["SF6"] * 3
    ds["Processing of SF6"] = record_step(
        ds["SF6"],
        previous=[(ds["SF6"] / 3, ds["Processing of SF6"])],
        parents=[ds["Processing of CO2"]],
        function="triple",
        description="tripled from CO2",
    )
    return ds


def written_nodes(path) -> int:
    with xr.open_dataset(path, engine="h5netcdf") as raw:
        return raw.sizes.get("processing node", 0)


def test_netcdf_round_trip(opulent_processing_ds, tmp_path, caplog):
    ds = modified_ds(opulent_processing_ds)
    attrs = dict(ds.attrs)
    names = list(ds.variables)

    ds.pr.to_netcdf(tmp_path / "temp.nc")

    # the dataset itself is not modified
    assert ds.attrs == attrs
    assert list(ds.variables) == names
    nds = primap2.open_dataset(tmp_path / "temp.nc")
    nds.pr.ensure_valid()
    assert not caplog.records
    xr.testing.assert_identical(ds, nds)
    assert "processing_info_format" not in nds.attrs
    assert set(nds.variables) == set(names)
    node = nds["Processing of SF6"].values.flat[0]
    assert sorted(s.description.function for s in node.history()) == [
        "double",
        "random",
        "random",
        "triple",
    ]
    # shared between the variables
    assert node.parents[1] is nds["Processing of CO2"].values.flat[0]


def test_netcdf_variables_dtypes(opulent_processing_ds, tmp_path):
    opulent_processing_ds.pr.to_netcdf(tmp_path / "temp.nc")
    with xr.open_dataset(tmp_path / "temp.nc", engine="h5netcdf") as raw:
        assert raw["Processing of CO2"].dtype == np.int64
        assert raw["Processing of CO2"].dims == opulent_processing_ds["CO2"].dims
        assert (raw["Processing of CO2"].values >= 0).all()
        assert raw.attrs["processing_info_format"] == 2
        assert raw["processing node time"].dtype == np.dtype("datetime64[ns]")


def test_netcdf_shared_nodes(tmp_path):
    """Cells of one variable and of different variables refer to one node object."""
    time = TIME[:1]
    coords = {"area (ISO3)": ["COL", "ARG"], "time": time}
    base = ProcessingStep(desc("base"), np.datetime64("2000"))

    def variable(name: str) -> xr.DataArray:
        return xr.DataArray(
            [[1.0], [2.0]],
            dims=["area (ISO3)", "time"],
            coords=coords,
            attrs={"entity": name, "units": f"{name} Gg / year"},
        )

    # the same step description for two variables and two timeseries
    shared_description = desc("sum")
    ds = xr.Dataset({"CO2": variable("CO2"), "SF6": variable("SF6")}, attrs={"area": "area (ISO3)"})
    for name in ("CO2", "SF6"):
        nodes = np.empty((2, 1), dtype=object)
        for i in range(2):
            nodes[i, 0] = ProcessingStep(shared_description, np.datetime64("2000"), (base,))
        ds[f"Processing of {name}"] = xr.DataArray(
            nodes,
            dims=["area (ISO3)", "time"],
            coords=coords,
            attrs={"entity": f"Processing of {name}", "described_variable": name},
        )

    ds.pr.to_netcdf(tmp_path / "temp.nc")
    # 4 nodes of the sums (equal, but not identical) and the base node
    assert written_nodes(tmp_path / "temp.nc") == 5
    nds = primap2.open_dataset(tmp_path / "temp.nc")

    bases = {
        id(nds[f"Processing of {name}"].values[i, 0].parents[0])
        for name in ("CO2", "SF6")
        for i in range(2)
    }
    assert len(bases) == 1
    descriptions = {
        id(nds[f"Processing of {name}"].values[i, 0].description)
        for name in ("CO2", "SF6")
        for i in range(2)
    }
    assert len(descriptions) == 1
    assert nds["Processing of CO2"].values[0, 0].description.id == shared_description.id


def test_netcdf_unreachable_nodes_not_written(opulent_processing_ds, tmp_path):
    ds = modified_ds(opulent_processing_ds)
    ds.pr.to_netcdf(tmp_path / "all.nc")
    selected = ds.pr.loc[{"area": ["COL"], "time": ["2000", "2001"]}]
    expected = {
        id(node)
        for var in selected.data_vars
        if var.startswith("Processing of ")
        for cell in selected[var].values.flat
        if cell is not None
        for node in cell.history()
    }

    selected.pr.to_netcdf(tmp_path / "selected.nc")

    assert written_nodes(tmp_path / "selected.nc") == len(expected)
    assert written_nodes(tmp_path / "selected.nc") < written_nodes(tmp_path / "all.nc")
    xr.testing.assert_identical(selected, primap2.open_dataset(tmp_path / "selected.nc"))


def test_netcdf_none_and_coords_round_trip(minimal_ds, tmp_path):
    ds = minimal_ds[["CO2"]]
    descriptions = [
        ProcessingStepDescription(function="f", description="d"),
        ProcessingStepDescription(function="f", description="d", source="", variable="", coords={}),
        ProcessingStepDescription(
            function="g\u00e4\n",
            description="",
            source="s",
            variable="CO2",
            coords={"a": 1, "b": 1.5, "c": True, "d": "x", "e": 0, "f": -3, "g": None},
        ),
    ]
    nodes = np.full(ds["CO2"].shape, None, dtype=object)
    times = ds["time"].values
    for i, description in enumerate(descriptions):
        for t in range(len(times)):
            nodes[t, i, 0] = ProcessingStep(description, times[t])
    nodes[:, 3, 0] = [ProcessingStep(descriptions[0], t) for t in times]
    ds["Processing of CO2"] = ds["CO2"].copy(data=nodes).rename("Processing of CO2")
    ds["Processing of CO2"].attrs = {"entity": "Processing of CO2", "described_variable": "CO2"}

    ds.pr.to_netcdf(tmp_path / "temp.nc")
    nds = primap2.open_dataset(tmp_path / "temp.nc")

    for i, description in enumerate(descriptions):
        got = nds["Processing of CO2"].values[0, i, 0].description
        assert got == description
        assert got.id == description.id
        assert got.source == description.source
        assert got.variable == description.variable
        assert got.coords == description.coords
    coords = nds["Processing of CO2"].values[0, 2, 0].description.coords
    assert [type(x) for x in coords.values()] == [int, float, bool, str, int, int, type(None)]
    assert nds["Processing of CO2"].values[0, 0, 0].description.coords is None
    assert nds["Processing of CO2"].values[0, 1, 0].description.coords == {}


def test_netcdf_no_processing_nodes(opulent_processing_ds, tmp_path):
    ds = opulent_processing_ds
    for var in ("CO2", "SF6", "CH4", "population"):
        ds[var] = ds[var].where(False)
        ds[f"Processing of {var}"] = ds[f"Processing of {var}"].copy(
            data=np.full(ds[var].shape, None, dtype=object)
        )

    ds.pr.to_netcdf(tmp_path / "temp.nc")
    nds = primap2.open_dataset(tmp_path / "temp.nc")

    xr.testing.assert_identical(ds, nds)
    assert nds["Processing of CO2"].isnull().all()


def test_netcdf_nan_for_missing(opulent_processing_ds, tmp_path):
    ds = opulent_processing_ds
    ds["CO2"] = ds["CO2"].where(~ds["area (ISO3)"].isin(["ARG"]))
    ds["Processing of CO2"] = ds["Processing of CO2"].where(ds["CO2"].notnull().values, np.nan)

    ds.pr.to_netcdf(tmp_path / "temp.nc")
    nds = primap2.open_dataset(tmp_path / "temp.nc")

    nds.pr.ensure_valid()
    assert nds["Processing of CO2"].pr.loc[{"area": "ARG"}].isnull().all()
    assert nds["Processing of CO2"].pr.loc[{"area": "COL"}].notnull().all()


def test_netcdf_old_format(tmp_path):
    ds = xr.Dataset(
        {
            "CO2": xr.DataArray(
                [[1.0]],
                dims=["area (ISO3)", "time"],
                coords={"area (ISO3)": ["COL"], "time": TIME[:1]},
                attrs={"entity": "CO2", "units": "Gg CO2 / year"},
            ),
            "Processing of CO2": xr.DataArray(
                np.array([b"\x81\xa5steps\x90"], dtype="S"),
                dims=["area (ISO3)"],
                coords={"area (ISO3)": ["COL"]},
                attrs={"entity": "Processing of CO2", "described_variable": "CO2"},
            ),
        }
    )
    ds.to_netcdf(tmp_path / "old.nc", engine="h5netcdf")

    with pytest.raises(ValueError, match="unsupported format"):
        primap2.open_dataset(tmp_path / "old.nc")


def test_netcdf_missing_version(opulent_processing_ds, tmp_path):
    opulent_processing_ds.pr.to_netcdf(tmp_path / "temp.nc")
    with xr.open_dataset(tmp_path / "temp.nc", engine="h5netcdf") as raw:
        raw = raw.load()
    del raw.attrs["processing_info_format"]
    raw.to_netcdf(tmp_path / "noversion.nc", engine="h5netcdf")

    with pytest.raises(ValueError, match="unsupported format"):
        primap2.open_dataset(tmp_path / "noversion.nc")


# --- recorder and decorator


def test_processing_step_records_changes():
    ds = processing_ds()

    with ds.pr.processing_step(function="fill", description="filled", source="s") as s:
        assert not s.ds.pr.has_processing_info()
        s.ds = s.ds.fillna(0)

    result = s.result
    np.testing.assert_array_equal(result["CO2"], [[1, 0, 3], [1, 2, 3]])
    filled = node_at(result, "COL", "2001")
    assert filled.time == np.datetime64("2001")
    assert filled.parents == ()
    assert filled.description.function == "fill"
    assert filled.description.variable == "CO2"
    assert filled.description.coords == {"area (ISO3)": "COL"}
    # unchanged data points keep their node
    assert node_at(result, "COL", "2000") is node_at(ds, "COL", "2000")
    assert node_at(result, "ARG", "2001") is node_at(ds, "ARG", "2001")
    ensure_complete_processing_info(result)


def test_processing_step_changed_value_uses_old_value():
    ds = processing_ds()

    with ds.pr.processing_step(function="scale", description="scaled") as s:
        s.ds["CO2"] = s.ds["CO2"] * 2

    new = node_at(s.result, "ARG", "2001")
    assert new.parents == (node_at(ds, "ARG", "2001"),)
    assert new.parents[0] is node_at(ds, "ARG", "2001")
    assert (
        node_at(s.result, "ARG", "2000").description is node_at(s.result, "ARG", "2001").description
    )
    # the NaN stays without node
    assert node_at(s.result, "COL", "2001") is None


def test_processing_step_in_place_changes():
    ds = processing_ds()

    with ds.pr.processing_step(function="set", description="set") as s:
        s.ds["CO2"].loc[{"area (ISO3)": "ARG", "time": "2000"}] = 5.0

    assert node_at(s.result, "ARG", "2000").description.function == "set"
    assert node_at(s.result, "ARG", "2001") is node_at(ds, "ARG", "2001")
    # the input is not modified
    assert ds["CO2"].pr.loc[{"area": "ARG", "time": "2000"}].item() == 1.0


def test_processing_step_no_change_logged(caplog):
    ds = processing_ds()

    with ds.pr.processing_step(function="nothing", description="nothing") as s:
        pass

    xr.testing.assert_identical(s.result, ds)
    assert "No data changed in the processing step of 'nothing'" in caplog.text


def test_processing_step_exception():
    ds = processing_ds()

    with (
        pytest.raises(KeyError),
        ds.pr.processing_step(function="f", description="d") as s,
    ):
        raise KeyError("error")

    with pytest.raises(RuntimeError, match="only available after the with block"):
        _ = s.result


def test_processing_step_result_within_block():
    ds = processing_ds()

    with (
        ds.pr.processing_step(function="f", description="d") as s,
        pytest.raises(RuntimeError),
    ):
        _ = s.result


def test_processing_step_changed_dimensions():
    ds = processing_ds()

    with (
        pytest.raises(ValueError, match="Dimensions of 'CO2' changed"),
        ds.pr.processing_step(function="f", description="d") as s,
    ):
        s.ds = s.ds.sum("area (ISO3)")


def test_processing_step_selection():
    ds = processing_ds()

    with ds.pr.processing_step(function="f", description="d") as s:
        s.ds = s.ds.pr.loc[{"area": ["COL"]}]

    assert s.result["Processing of CO2"].shape == (1, 3)
    assert node_at(s.result, "COL", "2000") is node_at(ds, "COL", "2000")


def test_processing_step_new_timeseries():
    ds = processing_ds()

    # timeseries without data have no history
    with ds.pr.processing_step(function="f", description="d") as s:
        s.ds = s.ds.reindex({"area (ISO3)": ["COL", "ARG", "BOL"]})
    assert (s.result["Processing of CO2"].pr.loc[{"area": "BOL"}].values == None).all()  # noqa: E711

    # timeseries with data are created by the processing step
    with ds.pr.processing_step(function="f", description="d", source="s") as s:
        s.ds = s.ds.reindex({"area (ISO3)": ["COL", "ARG", "BOL"]}, fill_value=1.0)
    bol = node_at(s.result, "BOL", "2000")
    assert bol.description.function == "f"
    assert bol.parents == ()


def test_processing_step_new_variable():
    ds = processing_ds()

    with ds.pr.processing_step(function="f", description="d", source="s") as s:
        s.ds["CH4"] = s.ds["CO2"]

    node = node_at(s.result, "ARG", "2000", var="CH4")
    assert node.description.function == "f"
    assert node.parents == ()


def test_processing_step_incomplete_processing_info():
    ds = processing_ds()
    ds["Processing of CO2"].values[0, 0] = None

    with pytest.raises(ValueError, match="contain data, but have no processing information"):
        ds.pr.processing_step(function="f", description="d")

    ds = processing_ds()
    ds["CH4"] = ds["CO2"]
    with pytest.raises(ValueError, match="has processing information, but not for 'CH4'"):
        ds.pr.processing_step(function="f", description="d")


@primap2.records_processing_step("ds", description="filled with {value}", source="{source}")
def fill_value(ds: xr.Dataset, value: float = 0.0, source: str = "default") -> xr.Dataset:
    assert not ds.pr.has_processing_info()
    return ds.fillna(value)


def test_records_processing_step():
    ds = processing_ds()

    result = fill_value(ds, source="test")

    np.testing.assert_array_equal(result["CO2"], [[1, 0, 3], [1, 2, 3]])
    filled = node_at(result, "COL", "2001")
    assert filled.description.function == "fill_value"
    assert filled.description.description == "filled with 0.0"
    assert filled.description.source == "test"
    assert node_at(result, "ARG", "2001") is node_at(ds, "ARG", "2001")
    # the input is not modified
    assert np.isnan(ds["CO2"].pr.loc[{"area": "COL", "time": "2001"}].item())


def test_records_processing_step_without_processing_info():
    ds = processing_ds().pr.without_processing_info()

    result = fill_value(value=1.0, ds=ds)

    assert not result.pr.has_processing_info()
    assert result["CO2"].pr.loc[{"area": "COL", "time": "2001"}].item() == 1.0


def test_records_processing_step_invalid():
    with pytest.raises(ValueError, match="has no argument 'da'"):
        primap2.records_processing_step("da", description="d")(fill_value.__wrapped__)

    with pytest.raises(TypeError, match="has to be a Dataset, not DataArray"):
        fill_value(processing_ds()["CO2"])


# --- splitting and recombining


def test_processing_info_split_and_recombine(opulent_processing_ds):
    ds = opulent_processing_ds

    data = ds.pr.without_processing_info()
    processing = ds.pr.processing_info()

    assert not data.pr.has_processing_info()
    assert set(processing.data_vars) == set(data.data_vars)
    assert processing.attrs == ds.attrs
    assert processing["CO2"].identical(ds["Processing of CO2"].rename("CO2"))
    xr.testing.assert_identical(data.pr.with_processing_info(processing), ds)


def test_processing_info_without_processing_info():
    ds = processing_ds().pr.without_processing_info()
    assert len(ds.pr.processing_info().data_vars) == 0


def test_processing_info_select():
    ds = processing_ds()

    col = ds.pr.processing_info().pr.loc[{"area": "COL", "time": "2000"}]["CO2"]

    assert col.item() is node_at(ds, "COL", "2000")


def test_with_processing_info_selected_data():
    """The processing information is selected to the coordinates of the data, also time."""
    ds = processing_ds()

    data = ds.pr.without_processing_info().pr.loc[{"area": ["ARG"], "time": ["2001", "2002"]}]
    result = data.pr.with_processing_info(ds.pr.processing_info())

    xr.testing.assert_identical(result, ds.pr.loc[{"area": ["ARG"], "time": ["2001", "2002"]}])


def test_with_processing_info_invalid():
    ds = processing_ds()
    data = ds.pr.without_processing_info()
    processing = ds.pr.processing_info()

    with pytest.raises(ValueError, match="already has processing information"):
        ds.pr.with_processing_info(processing)
    with pytest.raises(ValueError, match="'CH4', which is not a data variable"):
        data.pr.with_processing_info(processing.rename({"CO2": "CH4"}))
    with pytest.raises(ValueError, match="has the dimensions"):
        data.pr.with_processing_info(processing.expand_dims(source=["RAND2020"]))
    with pytest.raises(ValueError, match="has the dimensions"):
        data.pr.with_processing_info(processing.isel(time=0, drop=True))
    with pytest.raises(ValueError, match="lacks coordinates of the data"):
        data.pr.with_processing_info(processing.pr.loc[{"area": ["COL"]}])
    with pytest.raises(ValueError, match="lacks coordinates of the data"):
        data.pr.with_processing_info(processing.pr.loc[{"time": ["2000"]}])


def test_with_processing_info_incomplete(opulent_processing_ds):
    data = opulent_processing_ds.pr.without_processing_info()
    processing = opulent_processing_ds.pr.processing_info()

    with pytest.raises(ValueError, match="but not for 'CO2'"):
        data.pr.with_processing_info(processing.drop_vars("CO2"))
    # without any processing information, the result has none either
    result = data.pr.with_processing_info(processing.drop_vars(list(processing)))
    assert not result.pr.has_processing_info()


def test_remove_processing_info_deprecated():
    ds = processing_ds()
    with pytest.warns(DeprecationWarning, match="use without_processing_info"):
        result = ds.pr.remove_processing_info()
    xr.testing.assert_identical(result, ds.pr.without_processing_info())


# --- replaces, missing inputs


def test_processing_step_replaces():
    ds = processing_ds()

    with ds.pr.processing_step(function="set", description="set", source="s", replaces=True) as s:
        s.ds["CO2"] = s.ds["CO2"] * 2

    new = node_at(s.result, "ARG", "2001")
    assert new.parents == ()
    assert new.description.source == "s"
    # the NaN stays without node
    assert node_at(s.result, "COL", "2001") is None


def test_processing_step_without_inputs():
    ds = processing_ds()

    with (
        pytest.raises(ValueError, match="1 values of 'CO2' were set without inputs"),
        ds.pr.processing_step(function="fill", description="filled") as s,
    ):
        s.ds = s.ds.fillna(0)

    # replacing a value without a source leaves the new value without inputs
    with (
        pytest.raises(
            ValueError, match=re.escape("were set without inputs, e.g. area (ISO3)='COL'")
        ),
        ds.pr.processing_step(function="fill", description="filled", replaces=True) as s,
    ):
        s.ds = s.ds * 2
    # a value computed from the old value has an input
    with ds.pr.processing_step(function="scale", description="scaled") as s:
        s.ds = s.ds * 2


def test_processing_step_other_is_input():
    ds = processing_ds()
    other = ds["CO2"].fillna(5.0).to_dataset()
    other = xr.Dataset({"CO2": other["CO2"]}, attrs=ds.attrs)
    other = other["CO2"].pr.with_creation_step(function="read", description="read", source="o")

    with ds.pr.processing_step(function="fill", description="filled", other_ds=other) as s:
        s.ds = s.ds.fillna(other.pr.without_processing_info())

    assert node_at(s.result, "COL", "2001").parents == (node_at(other, "COL", "2001"),)


@primap2.records_processing_step("ds", description="replaced", source="s", replaces=True)
def replace_all(ds: xr.Dataset) -> xr.Dataset:
    return ds * 2


@primap2.records_processing_step("ds", description="no inputs", replaces=True)
def replace_without_source(ds: xr.Dataset) -> xr.Dataset:
    return ds * 2


def test_records_processing_step_replaces():
    ds = processing_ds()

    result = replace_all(ds)

    assert node_at(result, "ARG", "2001").parents == ()
    assert node_at(result, "ARG", "2001").description.source == "s"
    assert node_at(fill_value(ds), "COL", "2001").parents == ()


def test_records_processing_step_without_inputs():
    with pytest.raises(ValueError, match="were set without inputs"):
        replace_without_source(processing_ds())


def test_record_step_without_inputs():
    old, old_infos = previous_state()
    new = old * 2

    with pytest.raises(ValueError, match="4 values of 'CO2' were set without inputs"):
        record_step(new, previous=[(old, old_infos)], replaces=True, function="f", description="d")
    with pytest.raises(ValueError, match=re.escape("e.g. area='COL', time='2000'")):
        record_step(new, function="f", description="d")
    # a source or a parent is enough
    record_step(new, function="f", description="d", source="s")
    record_step(new, parents=[old_infos.isel(time=0, drop=True)], function="f", description="d")


def test_with_creation_step_without_source():
    da = xr.DataArray(
        [1.0, 2.0], dims=["time"], coords={"time": TIME[:2]}, name="CO2", attrs={"entity": "CO2"}
    )
    ds = da.pr.with_creation_step(function="read", description="read")
    assert ds["Processing of CO2"].values[0].parents == ()
    assert ds["Processing of CO2"].values[0].description.source is None

    ds = xr.Dataset({"CO2": da}).pr.with_creation_step(function="read", description="read")
    assert ds["Processing of CO2"].values[1].description.source is None


# --- fill, combine, merge, set


def other_ds() -> xr.Dataset:
    """Data for all years and areas, and an additional area, created by a leaf step."""
    da = xr.DataArray(
        [[1.0, 11.0, 3.0], [1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
        dims=["area (ISO3)", "time"],
        coords={"area (ISO3)": ["COL", "ARG", "MEX"], "time": TIME},
        attrs={"entity": "CO2", "units": "Gg CO2 / year"},
        name="CO2",
    )
    return xr.Dataset({"CO2": da}, attrs={"area": "area (ISO3)"}).pr.with_creation_step(
        function="read", description="read", source="other"
    )


@pytest.mark.parametrize("function", ["fillna", "combine_first", "merge"])
def test_fill_combine_merge_processing_info(function):
    ds = processing_ds()
    other = other_ds()

    result = getattr(ds.pr, function)(other)

    # a gap is filled from the other node at the same point, there is no old node
    filled = node_at(result, "COL", "2001")
    assert len(filled.parents) == 1
    assert filled.parents[0] is node_at(other, "COL", "2001")
    assert filled.description.function == function
    assert filled.time == np.datetime64("2001")
    # existing values keep their nodes, new timeseries are taken from other unchanged
    assert node_at(result, "COL", "2000") is node_at(ds, "COL", "2000")
    assert node_at(result, "ARG", "2001") is node_at(ds, "ARG", "2001")
    if function != "fillna":
        assert node_at(result, "MEX", "2001") is node_at(other, "MEX", "2001")
    ensure_complete_processing_info(result)


def test_fillna_value_processing_info():
    result = processing_ds().pr.fillna(0)

    filled = node_at(result, "COL", "2001")
    assert filled.parents == ()
    assert filled.description.source == "constant(0)"


def test_fillna_requires_processing_info_of_other():
    with pytest.raises(ValueError, match="has none"):
        processing_ds().pr.fillna(other_ds().pr.without_processing_info())


def test_set_processing_info():
    ds = processing_ds()
    other = other_ds().pr.loc[{"area": ["COL"]}]
    other["CO2"] = other["CO2"] * 2

    result = ds.pr.set("area", ["COL"], other, existing="overwrite")

    # overwritten values replace the old values: the old node is no parent
    for year in ["2000", "2001", "2002"]:
        new = node_at(result, "COL", year)
        assert len(new.parents) == 1
        assert new.parents[0] is node_at(other, "COL", year)
        assert new.description.function == "set"
    assert node_at(result, "ARG", "2000") is node_at(ds, "ARG", "2000")
    ensure_complete_processing_info(result)

    # a new timeseries is taken from the other dataset
    new_series = other_ds().pr.loc[{"area": ["MEX"]}]
    result = ds.pr.set("area", ["MEX"], new_series)
    assert node_at(result, "MEX", "2001") is node_at(new_series, "MEX", "2001")


def test_used_sources_leaves():
    result = processing_ds(sources=True).pr.used_sources()

    da = result["CO2"]
    assert da.dtype == np.int8
    assert da.dims == ("area (ISO3)", "time", "used source")
    assert list(da["used source"].values) == ["a", "b"]
    assert result.attrs == {"area": "area (ISO3)"}
    assert "flag_values" not in da.attrs
    # no data, no source
    assert da.sel({"area (ISO3)": "COL"}).values.tolist() == [[1, 0], [0, 0], [1, 0]]
    assert da.sel({"area (ISO3)": "ARG"}).values.tolist() == [[0, 1]] * 3


def test_used_sources_fillna():
    ds = processing_ds(sources=True)
    ds["Processing of CO2"] = leaves(ds["CO2"], {"COL": "own", "ARG": "own"}, sources=True)

    result = ds.pr.fillna(other_ds()).pr.used_sources()["CO2"]

    col = result.sel({"area (ISO3)": "COL"})
    assert list(col["used source"].values) == ["other", "own"]
    # the filled value is from the other dataset, the others are unchanged
    assert col.values.tolist() == [[0, 1], [1, 0], [0, 1]]
    assert result.sel({"area (ISO3)": "ARG"}).values.tolist() == [[0, 1]] * 3


def test_used_sources_aggregation():
    result = processing_ds(sources=True).pr.sum(dim="area").pr.used_sources()["CO2"]

    assert list(result["used source"].values) == ["a", "b"]
    # COL (a) has no data in 2001
    assert result.values.tolist() == [[1, 1], [0, 1], [1, 1]]


def test_used_sources_dim_name():
    result = processing_ds(sources=True).pr.used_sources(dim="src")

    assert result["CO2"].dims[-1] == "src"


def test_used_sources_leaf_without_source():
    ds = processing_ds()
    leaf = node_at(ds, "COL", "2000")

    with pytest.raises(ValueError, match=re.escape(str(leaf))):
        ds.pr.used_sources()


def test_used_sources_no_processing_info():
    with pytest.raises(ValueError, match="no processing information"):
        processing_ds().pr.without_processing_info().pr.used_sources()


def test_used_sources_shared_nodes_across_variables():
    ds = processing_ds(sources=True)
    ds["CH4"] = ds["CO2"] * 2
    ds["Processing of CH4"] = ds["Processing of CO2"].rename("Processing of CH4")
    ds["Processing of CH4"].attrs["described_variable"] = "CH4"
    ds["CH4"].attrs["entity"] = "CH4"

    result = ds.pr.used_sources()

    xr.testing.assert_equal(result["CO2"], result["CH4"])


def test_used_sources_many_sources():
    """More sources than the bits of one integer."""
    n = 130
    time = pd.date_range("2000", periods=3, freq="YS")
    da = xr.DataArray(
        np.ones((n, 3)),
        dims=["area", "time"],
        coords={"area": [f"A{i:03d}" for i in range(n)], "time": time},
        name="CO2",
    )
    descriptions = [desc(f"s{i}", f"S{i:03d}") for i in range(n)]
    nodes = np.array(
        [[ProcessingStep(d, t) for t in time.values] for d in descriptions], dtype=object
    )
    # the last timeseries builds on all others at the first time point
    nodes[-1, 0] = ProcessingStep(desc("all"), time.values[0], tuple(nodes[:-1, 0]))
    ds = xr.Dataset({"CO2": da}, attrs={"area": "area"})
    ds["Processing of CO2"] = xr.DataArray(
        nodes, dims=da.dims, coords=da.coords, attrs={"described_variable": "CO2"}
    )

    result = used_sources(ds)["CO2"]

    expected = np.broadcast_to(np.eye(n, dtype=np.int8)[:, np.newaxis], (n, 3, n)).copy()
    expected[-1, 0, :] = 1
    expected[-1, 0, -1] = 0
    assert (result.values == expected).all()


def test_used_sources_from_file(tmp_path):
    ds = processing_ds(sources=True)
    ds.pr.to_netcdf(tmp_path / "ds.nc")

    loaded = primap2.open_dataset(tmp_path / "ds.nc")

    xr.testing.assert_equal(loaded.pr.used_sources()["CO2"], ds.pr.used_sources()["CO2"])


def timeseries(name: str, years: range, *parents: ProcessingStep) -> list[ProcessingStep]:
    """Steps of one description for each year, each with the parents of the same year."""
    d = desc(name)
    return [
        ProcessingStep(d, np.datetime64(str(year)), tuple(p[year - years.start] for p in parents))
        for year in years
    ]


def test_format_timeseries_history_chain():
    a = timeseries("a", range(2000, 2004))
    b = timeseries("b", range(2000, 2004), a)

    assert primap2.format_timeseries_history(b) == (
        "b for 2000 to 2003: step b\n↑ a for 2000 to 2003: step a"
    )


def test_format_timeseries_history_branches():
    a = timeseries("a", range(2000, 2003))
    b = timeseries("b", range(2000, 2003), a)
    c = timeseries("c", range(2000, 2003), a)
    d = timeseries("d", range(2000, 2003), b, c)

    assert primap2.format_timeseries_history(d) == (
        "d for 2000 to 2002: step d\n"
        "├─ b for 2000 to 2002: step b\n"
        "│  ↑ (a) a for 2000 to 2002: step a\n"
        "└─ c for 2000 to 2002: step c\n"
        "   ↑ see (a)"
    )


def test_format_timeseries_history_other_time_points():
    fit = timeseries("fit", range(2000, 2003))
    extrapolated = ProcessingStep(desc("extrapolate"), np.datetime64("2005"), tuple(fit))
    early = [ProcessingStep(fit[0].description, np.datetime64(f"{y}")) for y in (1990, 1991)]

    assert primap2.format_timeseries_history([extrapolated]) == (
        "extrapolate for 2005: step extrapolate\n↑ fit for 2000 to 2002: step fit"
    )
    # steps of one description at several time points are grouped, with their parents
    other = ProcessingStep(extrapolated.description, np.datetime64("2007"), tuple(early))
    assert primap2.format_timeseries_history([extrapolated, other]) == (
        "extrapolate for 2005, 2007: step extrapolate\n↑ fit for 1990, 1991, 2000 to 2002: step fit"
    )


def test_format_timeseries_history_given_descriptions_reachable():
    a = timeseries("a", range(2000, 2003))
    b = timeseries("b", range(2000, 2003), a)

    # "a" is given as well and reached from "b"
    assert primap2.format_timeseries_history([*a, *b]) == (
        "(a) a for 2000 to 2002: step a\nb for 2000 to 2002: step b\n↑ see (a)"
    )


def test_format_timeseries_history_data_array():
    a = timeseries("a", range(2000, 2003))
    da = xr.DataArray(
        np.array([None, *a, np.nan], dtype=object),
        dims=["time"],
        coords={"time": pd.date_range("1999", periods=5, freq="YS")},
    )

    assert primap2.format_timeseries_history(da) == "a for 2000 to 2002: step a"
    assert primap2.format_timeseries_history([]) == ""


def test_format_timeseries_history_two_dimensional():
    with pytest.raises(ValueError, match="one timeseries"):
        primap2.format_timeseries_history(np.array([[None]], dtype=object))


def test_format_timeseries_history_processing_info():
    ds = processing_ds(sources=True).pr.fillna(other_ds())
    steps = ds.pr.processing_info()["CO2"].pr.loc[{"area": "COL"}]

    assert primap2.format_timeseries_history(steps) == (
        "a (data source: a) for 2000, 2002: step a\n"
        "CO2 area (ISO3)='COL': fillna for 2001: filled missing values\n"
        "↑ CO2 area (ISO3)='COL': read (data source: other) for 2001: read"
    )
