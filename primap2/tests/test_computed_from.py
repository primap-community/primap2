"""Tests for ProcessingStepRecorder.computed_from"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2
from primap2._downscale import _nearest_defined

AREA = "area (ISO3)"
CAT = "category (IPCC2006)"
TIMES = pd.date_range("2000", "2005", freq="YS")


def make_ds(values: dict, *, source: str = "S", cats=None, sources=("A",)) -> xr.Dataset:
    """Dataset with processing information, every data point with data has its own step.

    ``values`` maps the area to the values over time, copied for each source/category.
    """
    areas = list(values)
    coords = {"time": TIMES, AREA: areas, "source": list(sources)}
    arr = np.array([values[a] for a in areas], dtype=float).T[:, :, None]
    arr = np.repeat(arr, len(sources), axis=2)
    dims = ["time", AREA, "source"]
    if cats is not None:
        coords[CAT] = cats
        dims.insert(2, CAT)
        arr = np.repeat(arr[:, :, None], len(cats), axis=2) * (1 + np.arange(len(cats)))[:, None]
    da = xr.DataArray(arr, dims=dims, coords=coords, name="CO2", attrs={"entity": "CO2"})
    ch4 = da.fillna(0.0).rename("CH4").assign_attrs(entity="CH4")
    return xr.Dataset({"CO2": da, "CH4": ch4}).pr.with_creation_step(
        function="create", description="created", source=source
    )


def step_at(ds: xr.Dataset, var="CO2", **sel):
    return ds.pr.processing_info()[var].pr.loc[sel].item()


def at(ds, area, year, var="CO2", **sel):
    return step_at(ds, var, **{AREA: area, "time": str(year), **sel})


def gap_ds() -> xr.Dataset:
    return make_ds({"COL": [1, np.nan, np.nan, 4, 5, 6], "ARG": [1, 2, 3, 4, 5, 6]})


def nearest_known(da: xr.DataArray) -> xr.DataArray:
    """The time points of the last and the next value with data, along "nearest"."""
    # not xarray's cumulative max, which gives integers for nanosecond times (pandas 2)
    return _nearest_defined(da.notnull())


def ffill_declaration(step, ds, var="CO2"):
    step.computed_from(var, at={"time": nearest_known(ds[var]).isel(nearest=0)})


def test_forward_fill():
    ds = gap_ds()
    with ds.pr.processing_step(function="ffill", description="filled") as step:
        step.ds["CO2"] = step.ds["CO2"].ffill("time")
        ffill_declaration(step, ds)
    result = step.result
    first = at(ds, "COL", 2000)
    for year in (2001, 2002):
        new = at(result, "COL", year)
        assert new.description.function == "ffill"
        assert new.parents == (first,)
        assert new.time == np.datetime64(f"{year}")
    # unchanged values keep their steps
    assert at(result, "COL", 2003) is at(ds, "COL", 2003)
    assert at(result, "ARG", 2001) is at(ds, "ARG", 2001)
    used = result.pr.used_sources()["CO2"]
    col = used.pr.loc[{AREA: "COL", "source": "A"}].sel(time="2002")
    assert col.sel({"used source": "S"}).item() == 1


def test_forward_fill_equals_manual_parents():
    ds = gap_ds()
    with ds.pr.processing_step(function="ffill", description="filled") as step:
        step.ds["CO2"] = step.ds["CO2"].ffill("time")
        ffill_declaration(step, ds)
    infos = ds.pr.processing_info()["CO2"]
    last = infos.copy(
        data=pd.DataFrame(infos.values.reshape(6, -1)).ffill().values.reshape(infos.shape)
    )
    manual = primap2._processing_info.record_step(
        step.ds["CO2"],
        previous=[(ds["CO2"], ds["Processing of CO2"])],
        parents=[last],
        function="ffill",
        description="filled",
    )
    got = step.result["Processing of CO2"]
    for area in ("COL", "ARG"):
        for year in range(2000, 2006):
            sel = {AREA: area, "time": str(year), "source": "A"}
            a, b = got.pr.loc[sel].item(), manual.pr.loc[sel].item()
            assert (a is None) == (b is None)
            if a is not None:
                assert a.time == b.time
                assert [p.time for p in a.parents] == [p.time for p in b.parents]


def test_linear_interpolation_two_parents():
    ds = gap_ds()
    with ds.pr.processing_step(function="interp", description="interpolated") as step:
        step.ds["CO2"] = step.ds["CO2"].interpolate_na("time")
        step.computed_from("CO2", at={"time": nearest_known(ds["CO2"])})
    for year in (2001, 2002):
        new = at(step.result, "COL", year)
        assert {x.time for x in new.parents} == {np.datetime64("2000"), np.datetime64("2003")}
        assert len(new.parents) == 2


def test_list_of_times_is_fit():
    ds = gap_ds()
    with ds.pr.processing_step(function="fit", description="fit") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        # time points without data are skipped
        step.computed_from("CO2", at={"time": TIMES})
    new = at(step.result, "COL", 2001)
    assert {x.time for x in new.parents} == {
        np.datetime64(f"{y}") for y in (2000, 2003, 2004, 2005)
    }
    # nothing else changed
    assert at(step.result, "ARG", 2001) is at(ds, "ARG", 2001)


def test_labels_per_time():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        labels = xr.DataArray([TIMES[5], TIMES[0]], dims="time", coords={"time": TIMES[1:3]})
        step.computed_from("CO2", at={"time": labels})
    assert [x.time for x in at(step.result, "COL", 2001).parents] == [np.datetime64("2005")]
    assert [x.time for x in at(step.result, "COL", 2002).parents] == [np.datetime64("2000")]


def test_other_variable():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", variable="CH4")
    new = at(step.result, "COL", 2001)
    assert new.parents == (at(ds, "COL", 2001, var="CH4"),)


def test_list_of_variables():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", at={"time": "2000"}, variable=["CO2", "CH4"])
    new = at(step.result, "COL", 2001)
    assert new.parents == (at(ds, "COL", 2000), at(ds, "COL", 2000, var="CH4"))


def test_dataset_parent_with_source_mapping():
    ds = make_ds({"COL": [1, np.nan, 3, 4, 5, 6]}, sources=("A", "B"))
    other = make_ds({"COL": [7, 7, 7, 7, 7, 7]}, source="O", sources=("C",))
    with ds.pr.processing_step(function="f", description="d", source="X") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        labels = xr.DataArray(["C"], dims="source", coords={"source": ["A"]})
        step.computed_from("CO2", at={"source": labels}, dataset=other)
    new = at(step.result, "COL", 2001, source="A")
    assert new.parents == (at(other, "COL", 2001, source="C"),)
    # source B has no labels: no parents, changed but needs a source
    assert at(step.result, "COL", 2001, source="B").parents == ()


def test_dataset_parent_unmapped_source_missing():
    ds = make_ds({"COL": [1, np.nan, 3, 4, 5, 6]})
    other = make_ds({"COL": [7] * 6}, sources=("C",))
    with (
        pytest.raises(ValueError, match=r"source.*'A'"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", dataset=other)


def test_changed_dimensions_all_values_new():
    ds = make_ds({"COL": [1, np.nan, 3, 4, 5, 6]}, sources=("A", "B"))
    with ds.pr.processing_step(function="sum", description="summed the sources") as step:
        step.ds = step.ds.sum("source", min_count=1)
        step.computed_from("CO2", at={"source": ["A", "B"]})
        step.computed_from("CH4", at={"source": ["A", "B"]})
    new = step_at(step.result, **{AREA: "COL", "time": "2000"})
    assert new.description.function == "sum"
    assert {id(p) for p in new.parents} == {id(at(ds, "COL", 2000, source=s)) for s in "AB"}
    # no data, no step
    assert step_at(step.result, **{AREA: "COL", "time": "2001"}) is None


def test_changed_dimensions_without_computed_from():
    ds = make_ds({"COL": [1, np.nan, 3, 4, 5, 6]}, sources=("A", "B"))
    with (
        pytest.raises(ValueError, match=r"Dimensions of 'CH4' changed.*computed_from"),
        ds.pr.processing_step(function="sum", description="summed the sources") as step,
    ):
        step.ds = step.ds.sum("source", min_count=1)
        step.computed_from("CO2", at={"source": ["A", "B"]})


def test_category_basket_sum():
    cats = ["0", "1", "2"]
    ds = make_ds({"COL": [1, 2, 3, 4, 5, 6]}, cats=cats)
    with ds.pr.processing_step(function="sum", description="sum") as step:
        da = step.ds["CO2"]
        step.ds["CO2"] = xr.where(da[CAT] == "0", da.sel({CAT: ["1", "2"]}).sum(CAT), da)
        labels = xr.DataArray([["1", "2"]], dims=[CAT, "summand"], coords={CAT: ["0"]})
        step.computed_from("CO2", at={CAT: labels})
    new = at(step.result, "COL", 2002, **{CAT: "0"})
    assert set(map(id, new.parents)) == {id(at(ds, "COL", 2002, **{CAT: c})) for c in cats}
    # category "1" unchanged
    assert at(step.result, "COL", 2002, **{CAT: "1"}) is at(ds, "COL", 2002, **{CAT: "1"})


def test_calls_accumulate_and_missing_labels():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", at={"time": [TIMES[0], pd.NaT, TIMES[3]]})
        step.computed_from("CO2", at={"time": TIMES[4]})
    new = at(step.result, "COL", 2001)
    assert [x.time for x in new.parents] == [np.datetime64(y) for y in ("2000", "2003", "2004")]


def test_lists_combine_and_shared_dimensions_match():
    cats = ["0", "1", "2"]
    ds = make_ds({"COL": [1, 2, 3, 4, 5, 6]}, cats=cats)

    def parents(labels):
        with ds.pr.processing_step(function="f", description="d", replaces=True) as step:
            step.ds["CO2"] = step.ds["CO2"].where(step.ds[CAT] != "0", 0.0)
            step.computed_from("CO2", at=labels)
        new = at(step.result, "COL", 2003, **{CAT: "0"})
        return {(x.description.coords[CAT], x.time) for x in new.parents}

    times = [np.datetime64("2000"), np.datetime64("2001")]
    assert parents({CAT: ["1", "2"], "time": times}) == {(c, t) for c in "12" for t in times}
    shared = {
        CAT: xr.DataArray(["1", "2"], dims="pair"),
        "time": xr.DataArray(times, dims="pair"),
    }
    assert parents(shared) == {("1", times[0]), ("2", times[1])}


def test_replaces_keeps_computed_from_parents():
    ds = gap_ds()
    old = at(ds, "COL", 2000)
    with ds.pr.processing_step(function="f", description="d", replaces=True) as step:
        step.ds["CO2"] = step.ds["CO2"].where(step.ds["CO2"].time != TIMES[0], 10.0)
        step.computed_from("CO2", at={"time": TIMES[3]})
    new = at(step.result, "COL", 2000)
    assert old not in new.parents
    assert [x.time for x in new.parents] == [np.datetime64("2003")]


def test_no_changes_no_lookup():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        step.computed_from("CO2", at={"time": "1990"})
    assert at(step.result, "COL", 2000) is at(ds, "COL", 2000)


def test_error_unknown_dim():
    ds = gap_ds()
    with (
        pytest.raises(ValueError, match=r"'foo' is not a dimension"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", at={"foo": 1})


def test_error_nested_list():
    ds = gap_ds()
    with (
        pytest.raises(ValueError, match=r"list of labels"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", at={"time": [TIMES[:2], TIMES[2:4]]})


def test_error_labels_without_coordinates():
    ds = gap_ds()
    with (
        pytest.raises(ValueError, match=r"coordinates along 'time'"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", at={"time": xr.DataArray(TIMES.values, dims="time")})


def test_error_after_block():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        pass
    with pytest.raises(RuntimeError):
        step.computed_from("CO2")


def test_error_var_missing_in_result():
    ds = gap_ds()
    with (
        pytest.raises(ValueError, match=r"NOPE"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.computed_from("NOPE", variable="CO2")


def test_error_parent_dataset_without_processing_info():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        with pytest.raises(ValueError, match="processing information"):
            step.computed_from("CO2", dataset=ds.pr.without_processing_info())
        with pytest.raises(ValueError, match="processing information"):
            step.computed_from("CO2", variable="NOPE")
        with pytest.raises(ValueError, match="processing information"):
            step.computed_from("CO2", variable=["CH4", "NOPE"])


def test_error_unmapped_parent_dim_missing_in_new():
    other = make_ds({"COL": [1, np.nan, 3, 4, 5, 6]})
    ds = (
        other.pr.without_processing_info()
        .isel(source=0, drop=True)
        .pr.with_creation_step(function="create", description="created", source="S")
    )
    with (
        pytest.raises(ValueError, match=r"dimension 'source'.*'at'"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", dataset=other)


def test_missing_parent_labels_only_needed_ones():
    ds = gap_ds()
    # 2001 and 2002 are changed, 2003 is not
    unused = xr.DataArray(
        [TIMES[0], TIMES[0], pd.Timestamp("1990")], dims="time", coords={"time": TIMES[1:4]}
    )
    with ds.pr.processing_step(function="f", description="d") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", at={"time": unused})
    assert [x.time for x in at(step.result, "COL", 2001).parents] == [np.datetime64("2000")]
    needed = unused.copy(data=[TIMES[0], pd.Timestamp("1990"), TIMES[0]])
    with (
        pytest.raises(ValueError, match=r"parent variable.*1990"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", at={"time": needed})


def test_missing_parent_time_for_unmapped_time():
    ds = gap_ds()
    other = ds.isel(time=slice(0, 1))
    with (
        pytest.raises(ValueError, match=r"time"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", dataset=other)
