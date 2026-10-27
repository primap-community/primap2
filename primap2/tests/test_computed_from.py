"""Tests for ProcessingStepRecorder.computed_from"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2

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


def ffill_declaration(step, ds, var="CO2"):
    p = lambda da: da.rename(time="parent time")  # noqa: E731
    known = ds[var].notnull()
    count = known.cumsum("time")
    step.computed_from(var, lambda t: p(known) & (p(count) == count.sel(time=t)))


def test_forward_fill_with_callable():
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


def test_linear_interpolation_two_conditions():
    ds = gap_ds()
    with ds.pr.processing_step(function="interp", description="interpolated") as step:
        step.ds["CO2"] = step.ds["CO2"].interpolate_na("time")
        p = lambda da: da.rename(time="parent time")  # noqa: E731
        known = ds["CO2"].notnull()
        count = known.cumsum("time")
        step.computed_from(
            "CO2",
            lambda t: (
                p(known) & ((p(count) == count.sel(time=t)) | (p(count) == count.sel(time=t) + 1))
            ),
        )
    for year in (2001, 2002):
        new = at(step.result, "COL", year)
        assert {x.time for x in new.parents} == {np.datetime64("2000"), np.datetime64("2003")}
        assert len(new.parents) == 2


def test_array_mask_without_time_is_fit():
    ds = gap_ds()
    with ds.pr.processing_step(function="fit", description="fit") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        known = ds["CO2"].notnull()
        step.computed_from("CO2", known.rename(time="parent time"))
    new = at(step.result, "COL", 2001)
    assert {x.time for x in new.parents} == {
        np.datetime64(f"{y}") for y in (2000, 2003, 2004, 2005)
    }
    # nothing else changed
    assert at(step.result, "ARG", 2001) is at(ds, "ARG", 2001)


def test_array_mask_with_time():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        mask = xr.DataArray(
            np.zeros((len(TIMES), len(TIMES)), dtype=bool),
            dims=["time", "parent time"],
            coords={"time": TIMES, "parent time": TIMES},
        )
        mask.loc[{"time": "2001", "parent time": "2005"}] = True
        mask.loc[{"time": "2002", "parent time": "2000"}] = True
        step.computed_from("CO2", mask)
    assert [x.time for x in at(step.result, "COL", 2001).parents] == [np.datetime64("2005")]
    assert [x.time for x in at(step.result, "COL", 2002).parents] == [np.datetime64("2000")]


def test_mask_none_other_variable():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", variable="CH4")
    new = at(step.result, "COL", 2001)
    assert new.parents == (at(ds, "COL", 2001, var="CH4"),)


def test_dataset_parent_with_source_mapping():
    ds = make_ds({"COL": [1, np.nan, 3, 4, 5, 6]}, sources=("A", "B"))
    other = make_ds({"COL": [7, 7, 7, 7, 7, 7]}, source="O", sources=("C",))
    with ds.pr.processing_step(function="f", description="d", source="X") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        mask = xr.DataArray(
            [[True]],
            dims=["source", "parent source"],
            coords={"source": ["A"], "parent source": ["C"]},
        )
        step.computed_from("CO2", mask, dataset=other)
    new = at(step.result, "COL", 2001, source="A")
    assert new.parents == (at(other, "COL", 2001, source="C"),)
    # source B has no mask entry: no parents, changed but needs a source
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


def test_category_basket_sum():
    cats = ["0", "1", "2"]
    ds = make_ds({"COL": [1, 2, 3, 4, 5, 6]}, cats=cats)
    with ds.pr.processing_step(function="sum", description="sum") as step:
        da = step.ds["CO2"]
        step.ds["CO2"] = xr.where(da[CAT] == "0", da.sel({CAT: ["1", "2"]}).sum(CAT), da)
        mask = xr.DataArray(
            [[False, True, True]],
            dims=[CAT, f"parent {CAT}"],
            coords={CAT: ["0"], f"parent {CAT}": cats},
        )
        step.computed_from("CO2", mask)
    new = at(step.result, "COL", 2002, **{CAT: "0"})
    assert set(map(id, new.parents)) == {id(at(ds, "COL", 2002, **{CAT: c})) for c in cats}
    # category "1" unchanged
    assert at(step.result, "COL", 2002, **{CAT: "1"}) is at(ds, "COL", 2002, **{CAT: "1"})


def test_calls_accumulate_and_weights():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        weights = xr.DataArray(
            [0.5, 0, np.nan, 2.0, 0, 0],
            dims="parent time",
            coords={"parent time": TIMES},
        )
        step.computed_from("CO2", weights)
        step.computed_from(
            "CO2",
            xr.DataArray(
                [False, False, False, False, True, False],
                dims="parent time",
                coords={"parent time": TIMES},
            ),
        )
    new = at(step.result, "COL", 2001)
    assert {x.time for x in new.parents} == {np.datetime64(y) for y in ("2000", "2003", "2004")}


def test_replaces_keeps_computed_from_parents():
    ds = gap_ds()
    old = at(ds, "COL", 2000)
    with ds.pr.processing_step(function="f", description="d", replaces=True) as step:
        step.ds["CO2"] = step.ds["CO2"].where(step.ds["CO2"].time != TIMES[0], 10.0)
        step.computed_from("CO2", xr.DataArray(True).expand_dims({"parent time": TIMES[3:4]}))
    new = at(step.result, "COL", 2000)
    assert old not in new.parents
    assert [x.time for x in new.parents] == [np.datetime64("2003")]


def test_callable_only_called_for_changed_times():
    ds = gap_ds()
    called = []
    with ds.pr.processing_step(function="f", description="d", source="X") as step:
        step.ds["CO2"] = step.ds["CO2"].ffill("time")

        def mask(t):
            called.append(t)
            return xr.DataArray(True)

        step.computed_from("CO2", mask)
    assert called == [np.datetime64("2001"), np.datetime64("2002")]


def test_no_changes_no_calls():
    ds = gap_ds()
    with ds.pr.processing_step(function="f", description="d") as step:
        step.computed_from("CO2", lambda t: 1 / 0)
    assert at(step.result, "COL", 2000) is at(ds, "COL", 2000)


def test_error_unknown_mask_dim():
    ds = gap_ds()
    with (
        pytest.raises(ValueError, match=r"foo"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", xr.DataArray([True], dims="foo"))


def test_error_parent_dim_not_in_parent():
    ds = gap_ds()
    with (
        pytest.raises(ValueError, match=r"parent foo"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", xr.DataArray([True], dims="parent foo"))


def test_error_callable_with_time():
    ds = gap_ds()
    with (
        pytest.raises(ValueError, match=r"time"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", lambda t: xr.DataArray([True], dims="time", coords={"time": [t]}))


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


def test_error_unmapped_parent_dim_missing_in_new():
    other = make_ds({"COL": [1, np.nan, 3, 4, 5, 6]})
    ds = (
        other.pr.without_processing_info()
        .isel(source=0, drop=True)
        .pr.with_creation_step(function="create", description="created", source="S")
    )
    with (
        pytest.raises(ValueError, match=r"parent source"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", dataset=other)


def test_missing_parent_labels_only_needed_ones():
    ds = gap_ds()
    labels = [*TIMES[:2], pd.Timestamp("1990")]
    both = xr.DataArray([True, False, True], dims="parent time", coords={"parent time": labels})
    unused = xr.DataArray([True, False, False], dims="parent time", coords={"parent time": labels})
    with ds.pr.processing_step(function="f", description="d") as step:
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", unused)
    assert [x.time for x in at(step.result, "COL", 2001).parents] == [np.datetime64("2000")]
    with (
        pytest.raises(ValueError, match=r"parent variable.*1990"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", both)


def test_missing_parent_time_for_unmapped_time():
    ds = gap_ds()
    other = ds.isel(time=slice(0, 1))
    with (
        pytest.raises(ValueError, match=r"time"),
        ds.pr.processing_step(function="f", description="d") as step,
    ):
        step.ds["CO2"] = step.ds["CO2"].fillna(0.0)
        step.computed_from("CO2", dataset=other)
