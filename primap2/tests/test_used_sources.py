"""Verify the sources used per time point after functions which change data.

Every input data point has its own source, so that ``used_sources`` reveals which input
data points each output value was computed from. The expectations are derived from the
numerics of the functions, not from what the processing code records.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2
import primap2.csg
from primap2 import ProcessingStep, ProcessingStepDescription

AREA = "area (ISO3)"
CAT = "category (IPCC 2006)"
SRC = "source"
NAN = np.nan


# helpers


def years(first: int, last: int) -> pd.DatetimeIndex:
    return pd.date_range(str(first), str(last), freq="YS")


def make_da(
    name: str,
    data,
    coords: dict,
    *,
    entity: str | None = None,
    units: str = "Gg CO2 / year",
    gwp_context: str | None = None,
) -> xr.DataArray:
    attrs = {"entity": entity or name, "units": units}
    if gwp_context is not None:
        attrs["gwp_context"] = gwp_context
    return xr.DataArray(
        np.asarray(data, dtype=float), dims=list(coords), coords=coords, name=name, attrs=attrs
    )


def make_ds(*das: xr.DataArray, quantify: bool = True) -> xr.Dataset:
    ds = xr.Dataset({da.name: da for da in das})
    dims = set(ds.dims)
    ds.attrs.update({k: v for k, v in {"area": AREA, "cat": CAT}.items() if v in dims})
    return ds.pr.quantify() if quantify else ds


def leaf(prefix: str, var: str, *labels, year: int) -> str:
    """The source of the creation node of a data point, see tracked_per_point."""
    return "/".join([f"{prefix}{var}", *map(str, labels), str(year)])


def tracked_per_point(ds: xr.Dataset, prefix: str = "") -> xr.Dataset:
    """Add processing information: every data point with data is created by its own step
    without parents, whose source names the variable, the coordinates and the year."""
    result = ds.copy()
    for var in ds.data_vars:
        da = ds[var]
        dims = list(da.dims)
        time_axis = dims.index("time")
        values = np.full(da.shape, None, dtype=object)
        has_data = da.notnull().values
        for index in np.argwhere(has_data):
            index = tuple(index)
            labels = [
                da[dim].values[i] for dim, i in zip(dims, index, strict=True) if dim != "time"
            ]
            time = da["time"].values[index[time_axis]]
            year = pd.Timestamp(time).year
            description = ProcessingStepDescription(
                function="create",
                description="input data point",
                source=leaf(prefix, var, *labels, year=year),
                variable=str(var),
            )
            values[index] = ProcessingStep(description, time)
        name = f"Processing of {var}"
        result[name] = xr.DataArray(
            values,
            dims=da.dims,
            coords=da.coords,
            attrs={"entity": name, "described_variable": var},
        )
    return result


def used(ds: xr.Dataset, var: str, sel: dict | None = None) -> dict[int, set[str]]:
    """The used sources of a timeseries as {year: set of sources}, for the years with data."""
    sel = sel or {}
    sources = ds.pr.used_sources()[var].sel(sel)
    data = ds[var].sel(sel).pr.dequantify()
    assert set(sources.dims) == {"time", "used source"}, sources.dims
    sources = sources.transpose("time", "used source")
    result = {}
    for i, time in enumerate(sources["time"].values):
        if np.isnan(data.values[i]):
            assert not sources.values[i].any(), f"sources without data at {time}"
            continue
        result[pd.Timestamp(time).year] = set(
            np.asarray(sources["used source"].values)[sources.values[i] == 1].tolist()
        )
    return result


def own(prefix: str, var: str, *labels, year_range) -> dict[int, set[str]]:
    return {y: {leaf(prefix, var, *labels, year=y)} for y in year_range}


# fill and combine


@pytest.fixture
def fill_pair():
    """CO2 with gaps and a second dataset with other gaps, 2000 to 2004."""
    t = years(2000, 2004)
    coords = {AREA: ["COL", "MEX"], "time": t}
    start = make_ds(
        make_da("CO2", [[1, NAN, 3, NAN, NAN], [NAN, 2, NAN, 4, 5]], coords),
    )
    other = make_ds(
        make_da("CO2", [[10, 20, 30, 40, NAN], [10, 20, 30, NAN, 50]], coords),
    )
    return tracked_per_point(start), tracked_per_point(other, "o:")


def test_fillna_other(fill_pair):
    start, other = fill_pair
    result = start.pr.fillna(other)
    assert used(result, "CO2", {AREA: "COL"}) == {
        2000: {"CO2/COL/2000"},
        2001: {"o:CO2/COL/2001"},
        2002: {"CO2/COL/2002"},
        2003: {"o:CO2/COL/2003"},
    }
    assert used(result, "CO2", {AREA: "MEX"}) == {
        2000: {"o:CO2/MEX/2000"},
        2001: {"CO2/MEX/2001"},
        2002: {"o:CO2/MEX/2002"},
        2003: {"CO2/MEX/2003"},
        2004: {"CO2/MEX/2004"},
    }


def test_fillna_constant(fill_pair):
    start, _ = fill_pair
    result = start.pr.fillna(0)
    assert used(result, "CO2", {AREA: "COL"}) == {
        2000: {"CO2/COL/2000"},
        2001: {"constant(0)"},
        2002: {"CO2/COL/2002"},
        2003: {"constant(0)"},
        2004: {"constant(0)"},
    }
    assert used(result, "CO2", {AREA: "MEX"}) == {
        2000: {"constant(0)"},
        2001: {"CO2/MEX/2001"},
        2002: {"constant(0)"},
        2003: {"CO2/MEX/2003"},
        2004: {"CO2/MEX/2004"},
    }


def test_combine_first_extends_dimensions():
    """The other dataset has an additional area and additional years."""
    start = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                [[1, NAN, 3], [NAN, 2, NAN]],
                {AREA: ["COL", "MEX"], "time": years(2000, 2002)},
            )
        )
    )
    other = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                [[10, 20, 30, 40], [10, NAN, 30, 40], [10, 20, 30, NAN]],
                {AREA: ["COL", "MEX", "ARG"], "time": years(2000, 2003)},
            )
        ),
        "o:",
    )
    result = start.pr.combine_first(other)
    assert used(result, "CO2", {AREA: "COL"}) == {
        2000: {"CO2/COL/2000"},
        2001: {"o:CO2/COL/2001"},
        2002: {"CO2/COL/2002"},
        2003: {"o:CO2/COL/2003"},
    }
    assert used(result, "CO2", {AREA: "MEX"}) == {
        2000: {"o:CO2/MEX/2000"},
        2001: {"CO2/MEX/2001"},
        2002: {"o:CO2/MEX/2002"},
        2003: {"o:CO2/MEX/2003"},
    }
    # a timeseries only in other is taken as it is
    assert used(result, "CO2", {AREA: "ARG"}) == own(
        "o:", "CO2", "ARG", year_range=[2000, 2001, 2002]
    )


def test_merge_no_conflict(fill_pair):
    start, other = fill_pair
    # the overlap has to agree exactly for xarray's merge, take it from start
    other = other.copy(deep=True)
    overlap_free = other["CO2"].isnull() | start["CO2"].isnull()
    other["CO2"] = other["CO2"].where(overlap_free, start["CO2"])
    result = start.pr.merge(other)
    assert used(result, "CO2", {AREA: "COL"}) == {
        2000: {"CO2/COL/2000"},
        2001: {"o:CO2/COL/2001"},
        2002: {"CO2/COL/2002"},
        2003: {"o:CO2/COL/2003"},
    }
    assert used(result, "CO2", {AREA: "MEX"}) == {
        2000: {"o:CO2/MEX/2000"},
        2001: {"CO2/MEX/2001"},
        2002: {"o:CO2/MEX/2002"},
        2003: {"CO2/MEX/2003"},
        2004: {"CO2/MEX/2004"},
    }


def test_merge_within_tolerance_keeps_values_of_caller(fill_pair):
    """Where both have data and differ slightly, the value of the calling object wins."""
    start, other = fill_pair
    other = other.copy(deep=True)
    overlap_free = other["CO2"].isnull() | start["CO2"].isnull()
    other["CO2"] = other["CO2"].where(overlap_free, start["CO2"] * 1.005)
    result = start.pr.merge(other)
    np.testing.assert_allclose(
        result["CO2"].pint.magnitude[0, :3], [1.0, other["CO2"].pint.magnitude[0, 1], 3.0]
    )
    assert used(result, "CO2", {AREA: "COL"}) == {
        2000: {"CO2/COL/2000"},
        2001: {"o:CO2/COL/2001"},
        2002: {"CO2/COL/2002"},
        2003: {"o:CO2/COL/2003"},
    }
    assert used(result, "CO2", {AREA: "MEX"}) == {
        2000: {"o:CO2/MEX/2000"},
        2001: {"CO2/MEX/2001"},
        2002: {"o:CO2/MEX/2002"},
        2003: {"CO2/MEX/2003"},
        2004: {"CO2/MEX/2004"},
    }


def test_merge_dataarray_not_supported_without_processing_info(fill_pair):
    start, other = fill_pair
    with pytest.raises(TypeError):
        start.pr.merge(other["CO2"])


@pytest.fixture
def set_pair():
    """CO2 in COL (full data) and MEX (given per test), and values for MEX."""
    t = years(2000, 2003)

    def start(mex):
        return tracked_per_point(
            make_ds(
                make_da(
                    "CO2",
                    [[1, 2, 3, 4], mex],
                    {AREA: ["COL", "MEX"], "time": t},
                )
            )
        )

    value = tracked_per_point(
        make_ds(make_da("CO2", [[10, 20, NAN, 40]], {AREA: ["MEX"], "time": t})), "v:"
    )
    return start, value


def test_set_fillna_empty(set_pair):
    start, value = set_pair
    result = start([NAN] * 4).pr.set(AREA, "MEX", value)
    assert used(result, "CO2", {AREA: "MEX"}) == {
        2000: {"v:CO2/MEX/2000"},
        2001: {"v:CO2/MEX/2001"},
        2003: {"v:CO2/MEX/2003"},
    }
    assert used(result, "CO2", {AREA: "COL"}) == own("", "CO2", "COL", year_range=range(2000, 2004))


def test_set_fillna_empty_new_key(set_pair):
    start, value = set_pair
    result = start([NAN] * 4).pr.set(AREA, "ARG", value.assign_coords({AREA: ["ARG"]}))
    assert used(result, "CO2", {AREA: "ARG"}) == {
        2000: {"v:CO2/MEX/2000"},
        2001: {"v:CO2/MEX/2001"},
        2003: {"v:CO2/MEX/2003"},
    }
    assert used(result, "CO2", {AREA: "COL"}) == own("", "CO2", "COL", year_range=range(2000, 2004))


def test_set_fillna(set_pair):
    start, value = set_pair
    result = start([5, NAN, 7, NAN]).pr.set(AREA, "MEX", value, existing="fillna")
    assert used(result, "CO2", {AREA: "MEX"}) == {
        2000: {"CO2/MEX/2000"},
        2001: {"v:CO2/MEX/2001"},
        2002: {"CO2/MEX/2002"},
        2003: {"v:CO2/MEX/2003"},
    }


def test_set_overwrite_drops_the_old_source(set_pair):
    start, value = set_pair
    result = start([5, NAN, 7, 8]).pr.set(AREA, "MEX", value, existing="overwrite")
    # 2002 of the value is NaN, which overwrites the existing value
    assert used(result, "CO2", {AREA: "MEX"}) == {
        2000: {"v:CO2/MEX/2000"},
        2001: {"v:CO2/MEX/2001"},
        2003: {"v:CO2/MEX/2003"},
    }
    assert used(result, "CO2", {AREA: "COL"}) == own("", "CO2", "COL", year_range=range(2000, 2004))


# aggregation


def test_sum_over_area():
    t = years(2000, 2004)
    ds = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                [[1, NAN, 3, NAN, NAN], [NAN, 2, 3, NAN, 5], [1, NAN, NAN, NAN, 5]],
                {AREA: ["COL", "MEX", "ARG"], "time": t},
            )
        )
    )
    result = ds.pr.sum(dim=AREA)
    assert used(result, "CO2") == {
        2000: {"CO2/COL/2000", "CO2/ARG/2000"},
        2001: {"CO2/MEX/2001"},
        2002: {"CO2/COL/2002", "CO2/MEX/2002"},
        2004: {"CO2/MEX/2004", "CO2/ARG/2004"},
    }


def test_sum_over_category_keeps_areas():
    t = years(2000, 2002)
    ds = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                [[[1, 2, NAN], [NAN, 3, NAN]], [[NAN, NAN, NAN], [4, NAN, 5]]],
                {AREA: ["COL", "MEX"], CAT: ["1", "2"], "time": t},
            )
        )
    )
    result = ds.pr.sum(dim=CAT)
    assert used(result, "CO2", {AREA: "COL"}) == {
        2000: {"CO2/COL/1/2000"},
        2001: {"CO2/COL/1/2001", "CO2/COL/2/2001"},
    }
    assert used(result, "CO2", {AREA: "MEX"}) == {
        2000: {"CO2/MEX/2/2000"},
        2002: {"CO2/MEX/2/2002"},
    }


def test_add_aggregates_coordinates():
    t = years(2000, 2003)
    ds = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                [[1, NAN, 3, NAN], [2, 2, NAN, NAN], [NAN, NAN, NAN, NAN]],
                {AREA: ["COL", "MEX", "WORLD"], "time": t},
            )
        )
    )
    # WORLD has an existing value in 2001 which is kept (within the tolerance)
    ds["CO2"].loc[{AREA: "WORLD", "time": "2001"}] = 2.01 * primap2.ureg("Gg CO2 / year")
    ds = tracked_per_point(ds.pr.without_processing_info())
    result = ds.pr.add_aggregates_coordinates({AREA: {"WORLD": ["COL", "MEX"]}})
    assert used(result, "CO2", {AREA: "WORLD"}) == {
        2000: {"CO2/COL/2000", "CO2/MEX/2000"},
        2001: {"CO2/WORLD/2001"},
        2002: {"CO2/COL/2002"},
    }
    assert used(result, "CO2", {AREA: "COL"}) == own("", "CO2", "COL", year_range=[2000, 2002])


def test_add_aggregates_coordinates_overwrite_existing_value_not_within_tolerance_raises():
    """Sanity check of the setup: a deviating existing aggregate is an error."""
    t = years(2000, 2001)
    ds = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                [[1, 2], [2, 2], [10, NAN]],
                {AREA: ["COL", "MEX", "WORLD"], "time": t},
            )
        )
    )
    with pytest.raises(ValueError, match="discrepancies larger than tolerance"):
        ds.pr.add_aggregates_coordinates({AREA: {"WORLD": ["COL", "MEX"]}})


def test_sum_over_several_dimensions():
    ds = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                [[[1, 2, NAN], [NAN, 3, NAN]], [[NAN, NAN, NAN], [4, NAN, 5]]],
                {AREA: ["COL", "MEX"], CAT: ["1", "2"], "time": years(2000, 2002)},
            )
        )
    )
    result = ds.pr.sum(dim=[AREA, CAT])
    assert used(result, "CO2") == {
        2000: {"CO2/COL/1/2000", "CO2/MEX/2/2000"},
        2001: {"CO2/COL/1/2001", "CO2/COL/2/2001"},
        2002: {"CO2/MEX/2/2002"},
    }


def test_add_aggregates_coordinates_chained_rules():
    ds = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                [[1, NAN, 3], [2, 2, NAN], [NAN, 5, 6], [NAN, NAN, NAN], [NAN, NAN, NAN]],
                {AREA: ["COL", "MEX", "ARG", "REG", "WORLD"], "time": years(2000, 2002)},
            )
        )
    )
    result = ds.pr.add_aggregates_coordinates(
        {AREA: {"REG": ["COL", "MEX"], "WORLD": ["REG", "ARG"]}}
    )
    assert used(result, "CO2", {AREA: "REG"}) == {
        2000: {"CO2/COL/2000", "CO2/MEX/2000"},
        2001: {"CO2/MEX/2001"},
        2002: {"CO2/COL/2002"},
    }
    assert used(result, "CO2", {AREA: "WORLD"}) == {
        2000: {"CO2/COL/2000", "CO2/MEX/2000"},
        2001: {"CO2/MEX/2001", "CO2/ARG/2001"},
        2002: {"CO2/COL/2002", "CO2/ARG/2002"},
    }


def gas_ds():
    t = years(2000, 2004)
    coords = {AREA: ["COL", "MEX"], "time": t}
    return make_ds(
        make_da(
            "CO2",
            [[1, NAN, 3, NAN, NAN], [NAN, 2, NAN, 4, 5]],
            coords,
            units="Gg CO2 / year",
        ),
        make_da(
            "CH4",
            [[1, 2, NAN, NAN, 5], [NAN, 2, NAN, 4, NAN]],
            coords,
            units="Gg CH4 / year",
        ),
    )


def test_add_aggregates_variables_new_basket():
    ds = tracked_per_point(gas_ds())
    result = ds.pr.add_aggregates_variables({"KYOTOGHG (AR4GWP100)": ["CO2", "CH4"]})
    basket = "KYOTOGHG (AR4GWP100)"
    assert used(result, basket, {AREA: "COL"}) == {
        2000: {"CO2/COL/2000", "CH4/COL/2000"},
        2001: {"CH4/COL/2001"},
        2002: {"CO2/COL/2002"},
        2004: {"CH4/COL/2004"},
    }
    assert used(result, basket, {AREA: "MEX"}) == {
        2001: {"CO2/MEX/2001", "CH4/MEX/2001"},
        2003: {"CO2/MEX/2003", "CH4/MEX/2003"},
        2004: {"CO2/MEX/2004"},
    }
    # the contents are unchanged
    assert used(result, "CO2", {AREA: "COL"}) == own("", "CO2", "COL", year_range=[2000, 2002])


def test_add_aggregates_variables_existing_basket_keeps_its_values():
    ds = gas_ds()
    basket = "KYOTOGHG (AR4GWP100)"
    ds = ds.pr.add_aggregates_variables({basket: ["CO2", "CH4"]})
    # a basket which is consistent with the contents, but has only some values and
    # slightly different ones
    existing = ds[basket].copy()
    existing.pint.magnitude[...] = np.nan
    existing.loc[{AREA: "COL", "time": "2000"}] = (
        ds[basket].sel({AREA: "COL", "time": "2000"}) * 1.001
    )
    ds[basket] = existing
    ds = tracked_per_point(ds)
    result = ds.pr.add_aggregates_variables({basket: ["CO2", "CH4"]})
    assert used(result, basket, {AREA: "COL"}) == {
        2000: {f"{basket}/COL/2000"},
        2001: {"CH4/COL/2001"},
        2002: {"CO2/COL/2002"},
        2004: {"CH4/COL/2004"},
    }


def test_fill_all_na():
    t = years(2000, 2002)
    ds = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                [[1, NAN, 3], [NAN, NAN, NAN]],
                {AREA: ["COL", "MEX"], "time": t},
            )
        )
    )
    result = ds.pr.fill_all_na("time")
    assert used(result, "CO2", {AREA: "MEX"}) == {y: {"constant(0)"} for y in (2000, 2001, 2002)}
    assert used(result, "CO2", {AREA: "COL"}) == own("", "CO2", "COL", year_range=[2000, 2002])


# units


def test_unit_conversion_keeps_the_sources_per_point():
    ds = tracked_per_point(gas_ds())
    gwp = ds.pr.convert_to_gwp("AR4GWP100", "Gg CO2 / year")
    assert set(gwp.data_vars) >= {"CO2 (AR4GWP100)", "CH4 (AR4GWP100)"}
    assert used(gwp, "CH4 (AR4GWP100)", {AREA: "COL"}) == own(
        "", "CH4", "COL", year_range=[2000, 2001, 2004]
    )
    assert used(gwp, "CO2 (AR4GWP100)", {AREA: "MEX"}) == own(
        "", "CO2", "MEX", year_range=[2001, 2003, 2004]
    )
    mass = gwp.pr.convert_to_mass()
    assert set(mass.data_vars) >= {"CO2", "CH4"}
    assert used(mass, "CH4", {AREA: "COL"}) == own("", "CH4", "COL", year_range=[2000, 2001, 2004])
    assert used(mass, "CO2", {AREA: "MEX"}) == own("", "CO2", "MEX", year_range=[2001, 2003, 2004])


def test_quantify_dequantify_keep_the_sources():
    ds = tracked_per_point(gas_ds())
    expected = used(ds, "CO2", {AREA: "MEX"})
    assert used(ds.pr.dequantify(), "CO2", {AREA: "MEX"}) == expected
    assert used(ds.pr.dequantify().pr.quantify(), "CO2", {AREA: "MEX"}) == expected


# downscaling

KNOWN = [2000, 2005, 2010, 2015]
ALL_YEARS = range(1998, 2018)
CONTENTS = ["1.A", "1.B", "1.C"]


def basket_data(known_contents: np.ndarray) -> np.ndarray:
    """A basket for each year, which equals the sum of the contents where they are known;
    ``known_contents`` has the shape (contents, years)."""
    basket = 10.0 + np.arange(len(ALL_YEARS))
    sums = np.nansum(known_contents, axis=0)
    has = ~np.isnan(known_contents).all(axis=0)
    return np.where(has, sums, basket)


def contents_data(missing: dict[str, list[int]] | None = None) -> np.ndarray:
    """Three contents known at KNOWN years, (contents, years)."""
    missing = missing or {}
    data = np.full((3, len(ALL_YEARS)), NAN)
    for i, year in enumerate(KNOWN):
        for c, content in enumerate(CONTENTS):
            if year in missing.get(content, []):
                continue
            data[c, year - 1998] = (1.0 + i, 2.0, 3.0 + i)[c]
    return data


def category_ds(missing: dict[str, list[int]] | None = None, areas=("COL", "MEX")) -> xr.Dataset:
    contents = contents_data(missing)
    basket = basket_data(contents)
    per_area = []
    for k, _ in enumerate(areas):
        scale = 1.0 + k
        per_area.append(np.vstack([basket * scale, contents * scale]))
    return tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                np.stack(per_area),
                {AREA: list(areas), CAT: ["1", *CONTENTS], "time": years(1998, 2017)},
            )
        )
    )


def expected_downscaled(
    leaf_of, basket_leaf, defined: dict[str, list[int]], content: str, year: int
) -> set[str]:
    """Hand-derived: the basket at the year, and all contents with data at the nearest
    years before and after at which the share of this content was defined."""
    before = [y for y in defined[content] if y <= year]
    after = [y for y in defined[content] if y >= year]
    sources = {basket_leaf(year)}
    for y in (max(before) if before else None, min(after) if after else None):
        if y is None:
            continue
        sources |= {leaf_of(c, y) for c in CONTENTS if y in defined[c]}
    return sources


def check_downscaled(got: dict[int, set[str]], leaf_of, basket_leaf, defined, content: str) -> None:
    expected = {}
    for year in ALL_YEARS:
        if year in defined[content]:
            expected[year] = {leaf_of(content, year)}
        else:
            expected[year] = expected_downscaled(leaf_of, basket_leaf, defined, content, year)
    assert got == expected


def test_downscale_timeseries_several_gaps():
    ds = category_ds()
    result = ds.pr.downscale_timeseries(dim=CAT, basket="1", basket_contents=CONTENTS)
    area = "COL"

    def leaf_of(content, year):
        return f"CO2/{area}/{content}/{year}"

    def contents_at(*ys):
        return {leaf_of(c, y) for c in CONTENTS for y in ys}

    for content in CONTENTS:
        got = used(result, "CO2", {AREA: area, CAT: content})
        assert set(got) == set(ALL_YEARS)
        # explicit values from the task description
        assert got[2007] == {f"CO2/{area}/1/2007"} | contents_at(2005, 2010)
        assert got[1999] == {f"CO2/{area}/1/1999"} | contents_at(2000)
        assert got[1998] == {f"CO2/{area}/1/1998"} | contents_at(2000)
        assert got[2016] == {f"CO2/{area}/1/2016"} | contents_at(2015)
        assert got[2017] == {f"CO2/{area}/1/2017"} | contents_at(2015)
        for y in (2001, 2004):
            assert got[y] == {f"CO2/{area}/1/{y}"} | contents_at(2000, 2005)
        for y in (2006, 2009):
            assert got[y] == {f"CO2/{area}/1/{y}"} | contents_at(2005, 2010)
        for y in (2011, 2014):
            assert got[y] == {f"CO2/{area}/1/{y}"} | contents_at(2010, 2015)
        for y in KNOWN:
            assert got[y] == {leaf_of(content, y)}
    # the basket is unchanged
    assert used(result, "CO2", {AREA: "MEX", CAT: "1"}) == {
        y: {f"CO2/MEX/1/{y}"} for y in ALL_YEARS
    }
    # areas don't mix
    got = used(result, "CO2", {AREA: "MEX", CAT: "1.B"})
    assert got[2007] == {"CO2/MEX/1/2007"} | {
        f"CO2/MEX/{c}/{y}" for c in CONTENTS for y in (2005, 2010)
    }


def test_downscale_timeseries_content_missing_at_a_share_year():
    """1.C is not known in 2005: its shares are defined at the other years only, while
    the shares of 1.A and 1.B in 2005 are defined for the contents with data."""
    missing = {"1.C": [2005]}
    ds = category_ds(missing)
    result = ds.pr.downscale_timeseries(
        dim=CAT, basket="1", basket_contents=CONTENTS, check_consistency=False
    )
    area = "COL"
    defined = {c: [y for y in KNOWN if y not in missing.get(c, [])] for c in CONTENTS}

    def leaf_of(content, year):
        return f"CO2/{area}/{content}/{year}"

    def basket_leaf(year):
        return f"CO2/{area}/1/{year}"

    for content in CONTENTS:
        got = used(result, "CO2", {AREA: area, CAT: content})
        # 1.C is downscaled in 2005 from 2000 and 2010, the others keep their data
        if content == "1.C":
            expected_2005 = {basket_leaf(2005)} | {
                leaf_of(c, y) for c in CONTENTS for y in (2000, 2010)
            }
            assert got[2005] == expected_2005
        check_downscaled(got, leaf_of, basket_leaf, defined, content)
    # explicit: 1.A in 2003 uses all contents in 2000 and only 1.A and 1.B in 2005
    got = used(result, "CO2", {AREA: area, CAT: "1.A"})
    assert got[2003] == (
        {basket_leaf(2003)}
        | {leaf_of(c, 2000) for c in CONTENTS}
        | {leaf_of(c, 2005) for c in ("1.A", "1.B")}
    )
    # explicit: 1.C in 2003 skips 2005
    got = used(result, "CO2", {AREA: area, CAT: "1.C"})
    assert got[2003] == (
        {basket_leaf(2003)} | {leaf_of(c, y) for c in CONTENTS for y in (2000, 2010)}
    )


def varying_category_ds() -> xr.Dataset:
    """CO2 and CH4, the contents of each area and variable are known in other years."""
    known = {
        ("CO2", "COL"): [2000, 2005, 2010, 2015],
        ("CO2", "MEX"): [2002, 2008],
        ("CH4", "COL"): [1998, 2017],
        ("CH4", "MEX"): [2004, 2005, 2012],
    }
    das = []
    for var in ("CO2", "CH4"):
        data = np.full((2, 4, len(ALL_YEARS)), NAN)
        for a, area in enumerate(("COL", "MEX")):
            for y in known[var, area]:
                data[a, 1:, y - 1998] = [1.0 + y % 3, 2.0, 3.0 + y % 5]
                data[a, 0, y - 1998] = data[a, 1:, y - 1998].sum()
            for y in ALL_YEARS:
                if y not in known[var, area]:
                    data[a, 0, y - 1998] = 20.0 + y - 1998
        das.append(
            make_da(
                var,
                data,
                {AREA: ["COL", "MEX"], CAT: ["1", *CONTENTS], "time": years(1998, 2017)},
                units=f"Gg {var} / year",
            )
        )
    return tracked_per_point(make_ds(*das)), known


def test_downscale_timeseries_share_years_differ_per_area_and_variable():
    ds, known = varying_category_ds()
    result = ds.pr.downscale_timeseries(dim=CAT, basket="1", basket_contents=CONTENTS)
    for (var, area), defined_years in known.items():
        defined = dict.fromkeys(CONTENTS, defined_years)
        for content in CONTENTS:
            check_downscaled(
                used(result, var, {AREA: area, CAT: content}),
                lambda c, y, var=var, area=area: f"{var}/{area}/{c}/{y}",
                lambda y, var=var, area=area: f"{var}/{area}/1/{y}",
                defined,
                content,
            )


def test_downscale_timeseries_sel_only_changes_the_selection():
    ds = category_ds()
    result = ds.pr.downscale_timeseries(
        dim=CAT, basket="1", basket_contents=CONTENTS, sel={AREA: ["COL"]}
    )
    for content in CONTENTS:
        assert used(result, "CO2", {AREA: "COL", CAT: content}).keys() == set(ALL_YEARS)
        assert used(result, "CO2", {AREA: "MEX", CAT: content}) == own(
            "", "CO2", "MEX", content, year_range=KNOWN
        )
    got = used(result, "CO2", {AREA: "COL", CAT: "1.A"})
    assert got[2003] == {"CO2/COL/1/2003"} | {
        f"CO2/COL/{c}/{y}" for c in CONTENTS for y in (2000, 2005)
    }


def test_downscale_timeseries_contents_known_in_different_years():
    """Shares are defined for each content where it has data, from the contents which
    have data in that year."""
    defined = {"1.A": [2000, 2004, 2010], "1.B": [2002, 2004], "1.C": [2004, 2008]}
    data = np.full((1, 4, len(ALL_YEARS)), NAN)
    for c, content in enumerate(CONTENTS):
        for y in defined[content]:
            data[0, 1 + c, y - 1998] = 1.0 + c + (y % 4)
    basket = np.where(
        np.isnan(data[0, 1:]).all(axis=0),
        30.0 + np.arange(len(ALL_YEARS)),
        np.nansum(data[0, 1:], axis=0),
    )
    data[0, 0] = basket
    ds = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                data,
                {AREA: ["COL"], CAT: ["1", *CONTENTS], "time": years(1998, 2017)},
            )
        )
    )
    result = ds.pr.downscale_timeseries(dim=CAT, basket="1", basket_contents=CONTENTS)
    for content in CONTENTS:
        check_downscaled(
            used(result, "CO2", {AREA: "COL", CAT: content}),
            lambda c, y: f"CO2/COL/{c}/{y}",
            lambda y: f"CO2/COL/1/{y}",
            defined,
            content,
        )
    # explicit: 1.B in 2003 interpolates between its shares in 2002 (1.A, 1.B) and
    # 2004 (all contents)
    got = used(result, "CO2", {AREA: "COL", CAT: "1.B"})
    assert got[2003] == {
        "CO2/COL/1/2003",
        "CO2/COL/1.B/2002",
        "CO2/COL/1.A/2004",
        "CO2/COL/1.B/2004",
        "CO2/COL/1.C/2004",
    }
    # 1.B in 2002 is known, but 1.A in 2002 is downscaled from 2000 and 2004
    assert got[2002] == {"CO2/COL/1.B/2002"}
    # explicit: 1.C in 2001 uses its only neighbour, 2004
    got = used(result, "CO2", {AREA: "COL", CAT: "1.C"})
    assert got[2001] == {"CO2/COL/1/2001"} | {f"CO2/COL/{c}/2004" for c in CONTENTS}


def gas_downscaling_ds(missing: dict[str, list[int]] | None = None) -> xr.Dataset:
    contents = contents_data(missing)
    gwp = {"CO2": 1.0, "CH4": 25.0, "N2O": 298.0}
    names = list(gwp)
    t = years(1998, 2017)
    das = []
    for name, row in zip(names, contents, strict=True):
        data = np.stack([row, row * 2])
        das.append(
            make_da(
                name,
                data,
                {AREA: ["COL", "MEX"], "time": t},
                units=f"Gg {name} / year",
            )
        )
    co2e = np.nansum([contents[i] * gwp[n] for i, n in enumerate(names)], axis=0)
    has = ~np.isnan(contents).all(axis=0)
    basket = np.where(has, co2e, 5000.0 + 100 * np.arange(len(ALL_YEARS)))
    # an inconsistent basket where a content is missing is not checked
    das.append(
        make_da(
            "KYOTOGHG (AR4GWP100)",
            np.stack([basket, basket * 2]),
            {AREA: ["COL", "MEX"], "time": t},
            entity="KYOTOGHG",
            gwp_context="AR4GWP100",
        )
    )
    return tracked_per_point(make_ds(*das))


def test_downscale_gas_timeseries_several_gaps():
    ds = gas_downscaling_ds()
    basket = "KYOTOGHG (AR4GWP100)"
    gases = ["CO2", "CH4", "N2O"]
    result = ds.pr.downscale_gas_timeseries(basket=basket, basket_contents=gases)
    area = "COL"

    def contents_at(*ys):
        return {f"{g}/{area}/{y}" for g in gases for y in ys}

    for gas in gases:
        got = used(result, gas, {AREA: area})
        assert set(got) == set(ALL_YEARS)
        b = lambda y: f"{basket}/{area}/{y}"  # noqa: E731
        assert got[1999] == {b(1999)} | contents_at(2000)
        assert got[2007] == {b(2007)} | contents_at(2005, 2010)
        assert got[2013] == {b(2013)} | contents_at(2010, 2015)
        assert got[2016] == {b(2016)} | contents_at(2015)
        for y in (1998, 2001, 2004):
            assert got[y] == {b(y)} | contents_at(
                *{1998: (2000,), 2001: (2000, 2005), 2004: (2000, 2005)}[y]
            )
        for y in KNOWN:
            assert got[y] == {f"{gas}/{area}/{y}"}
    assert used(result, basket, {AREA: "MEX"}) == {y: {f"{basket}/MEX/{y}"} for y in ALL_YEARS}


def test_downscale_gas_timeseries_content_missing_at_a_share_year():
    missing = {"1.B": [2005]}
    ds = gas_downscaling_ds(missing)
    basket = "KYOTOGHG (AR4GWP100)"
    gases = ["CO2", "CH4", "N2O"]
    result = ds.pr.downscale_gas_timeseries(
        basket=basket, basket_contents=gases, check_consistency=False
    )
    area = "COL"
    defined = {
        "1.A": [y for y in KNOWN],
        "1.B": [y for y in KNOWN if y != 2005],
        "1.C": [y for y in KNOWN],
    }
    to_gas = dict(zip(CONTENTS, gases, strict=True))
    to_cat = {g: c for c, g in to_gas.items()}

    def leaf_of(content, year):
        return f"{to_gas[content]}/{area}/{year}"

    def basket_leaf(year):
        return f"{basket}/{area}/{year}"

    for gas in gases:
        got = used(result, gas, {AREA: area})
        check_downscaled(got, leaf_of, basket_leaf, defined, to_cat[gas])


def test_downscale_timeseries_by_shares():
    t = years(2000, 2003)
    contents = ["1.A", "1.B", "1.C"]
    data = np.full((2, 4, 4), NAN)
    data[0, 0] = [10, 11, 12, 13]
    data[1, 0] = [20, 21, 22, 23]
    ds = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                data,
                {AREA: ["COL", "MEX"], CAT: ["1", *contents], "time": t},
            )
        )
    )
    # 1.C has no share in 2002
    shares = make_ds(
        make_da(
            "CO2",
            [[1, 2, 3, 1], [2, 2, 2, 2], [1, 1, NAN, 3]],
            {CAT: contents, "time": t},
            units="",
        )
    )
    shares = tracked_per_point(shares, "s:")
    result = ds.pr.downscale_timeseries_by_shares(
        dim=CAT, basket="1", basket_contents=contents, basket_contents_shares=shares
    )
    for area in ("COL", "MEX"):

        def basket(y, area=area):
            return f"CO2/{area}/1/{y}"

        def shares_at(y, cs):
            return {f"s:CO2/{c}/{y}" for c in cs}

        for content in ("1.A", "1.B"):
            assert used(result, "CO2", {AREA: area, CAT: content}) == {
                2000: {basket(2000)} | shares_at(2000, contents),
                2001: {basket(2001)} | shares_at(2001, contents),
                2002: {basket(2002)} | shares_at(2002, ["1.A", "1.B"]),
                2003: {basket(2003)} | shares_at(2003, contents),
            }
        assert used(result, "CO2", {AREA: area, CAT: "1.C"}) == {
            2000: {basket(2000)} | shares_at(2000, contents),
            2001: {basket(2001)} | shares_at(2001, contents),
            2003: {basket(2003)} | shares_at(2003, contents),
        }
        assert used(result, "CO2", {AREA: area, CAT: "1"}) == {
            y: {basket(y)} for y in range(2000, 2004)
        }


# compose


def csg_input(sources: dict[str, list[float]], first=2000) -> xr.Dataset:
    n = len(next(iter(sources.values())))
    return tracked_per_point(
        make_ds(
            make_da(
                "CH4",
                np.array(list(sources.values()))[:, None, :],
                {SRC: list(sources), AREA: ["COL"], "time": years(first, first + n - 1)},
                units="Gg CH4 / year",
            )
        )
    )


def compose(ds, strategies, order=None):
    order = order or list(ds[SRC].values)
    return primap2.csg.compose(
        input_data=ds,
        priority_definition=primap2.csg.PriorityDefinition(
            priority_dimensions=[SRC], priorities=[{SRC: s} for s in order]
        ),
        strategy_definition=primap2.csg.StrategyDefinition(strategies=strategies),
        progress_bar=None,
    )


def test_compose_substitution():
    ds = csg_input(
        {
            "A": [1, NAN, NAN, 4, NAN, NAN],
            "B": [10, 20, NAN, 40, NAN, 60],
            "C": [100, 200, 300, NAN, NAN, 600],
        }
    )
    result = compose(ds, [({}, primap2.csg.SubstitutionStrategy())])
    assert used(result, "CH4", {AREA: "COL"}) == {
        2000: {"CH4/A/COL/2000"},
        2001: {"CH4/B/COL/2001"},
        2002: {"CH4/C/COL/2002"},
        2003: {"CH4/A/COL/2003"},
        2005: {"CH4/B/COL/2005"},
    }


def test_compose_global_least_squares():
    a = [1, 2, 3, NAN, NAN, NAN, 7, 8]
    b = [2, 4, 6, 8, 10, NAN, NAN, 16]
    c = [3, NAN, 9, 12, 15, 18, 21, NAN]
    ds = csg_input({"A": a, "B": b, "C": c})
    result = compose(
        ds,
        [
            ({SRC: "A"}, primap2.csg.SubstitutionStrategy()),
            ({}, primap2.csg.GlobalLSStrategy()),
        ],
    )

    def s(src, *ys):
        return {f"CH4/{src}/COL/{y}" for y in ys}

    got = used(result, "CH4", {AREA: "COL"})
    assert got[2000] == s("A", 2000)
    assert got[2006] == s("A", 2006)
    # B is matched to A on 2000, 2001, 2002 and 2007, the overlap of both
    fit_ab = s("A", 2000, 2001, 2002, 2007) | s("B", 2000, 2001, 2002, 2007)
    assert got[2003] == s("B", 2003) | fit_ab
    assert got[2004] == s("B", 2004) | fit_ab
    # C is matched to the result so far on 2000, 2002, 2003, 2004 and 2006
    result_at = {
        2000: s("A", 2000),
        2002: s("A", 2002),
        2003: s("B", 2003) | fit_ab,
        2004: s("B", 2004) | fit_ab,
        2006: s("A", 2006),
    }
    expected_2005 = s("C", 2005, 2000, 2002, 2003, 2004, 2006)
    for sources in result_at.values():
        expected_2005 |= sources
    assert got[2005] == expected_2005
    assert set(got) == set(range(2000, 2008))


def test_compose_local_trends_two_gaps():
    n = 20
    a = [float(i + 1) for i in range(n)]
    for i in (5, 6, 7, 13, 14, 15):
        a[i] = NAN
    b = [2.0 * (i + 1) + 1 for i in range(n)]
    ds = csg_input({"A": a, "B": b})
    fit_params = primap2.csg.FitParameters(
        fit_degree=1, trend_length=3, min_trend_points=3, fallback_degree=0
    )
    result = compose(
        ds,
        [
            ({SRC: "A"}, primap2.csg.SubstitutionStrategy()),
            ({}, primap2.csg.LocalTrendsStrategy(fit_params=fit_params)),
        ],
    )

    def both(*ys):
        return {f"CH4/{src}/COL/{y}" for src in "AB" for y in ys}

    got = used(result, "CH4", {AREA: "COL"})
    assert set(got) == set(range(2000, 2020))
    # gap 2005-2007 uses the windows 2002-2004 and 2008-2010 only
    for y in (2005, 2006, 2007):
        assert got[y] == {f"CH4/B/COL/{y}"} | both(2002, 2003, 2004, 2008, 2009, 2010)
    # gap 2013-2015 uses the windows 2010-2012 and 2016-2018 only
    for y in (2013, 2014, 2015):
        assert got[y] == {f"CH4/B/COL/{y}"} | both(2010, 2011, 2012, 2016, 2017, 2018)
    for y in (2000, 2004, 2008, 2012, 2019):
        assert got[y] == {f"CH4/A/COL/{y}"}


# decorator and processing_step


@primap2.records_processing_step("ds", function="change", description="changed", source="const")
def change_two_values(ds: xr.Dataset) -> xr.Dataset:
    """Sets COL 2001 (no data before) and COL 2002 (data before)."""
    ds = ds.copy(deep=True)
    ds["CO2"].loc[{AREA: "COL", "time": ["2001", "2002"]}] = 7 * primap2.ureg("Gg CO2 / year")
    return ds


@primap2.records_processing_step(
    "ds", other="other", function="take", description="taken from other"
)
def take_from_other(ds: xr.Dataset, other: xr.Dataset) -> xr.Dataset:
    """Overwrites COL 2002 (data before) and sets COL 2001 (no data before)."""
    ds = ds.copy(deep=True)
    ds["CO2"].loc[{AREA: "COL", "time": ["2001", "2002"]}] = other["CO2"].loc[
        {AREA: "COL", "time": ["2001", "2002"]}
    ]
    return ds


@primap2.records_processing_step(
    "ds", other="other", function="take", description="taken from other", replaces=True
)
def take_from_other_replacing(ds: xr.Dataset, other: xr.Dataset) -> xr.Dataset:
    return take_from_other.__wrapped__(ds, other)


@pytest.fixture
def step_pair(fill_pair):
    start, other = fill_pair
    return start, other


def test_decorator_default_old_value_is_used(step_pair):
    start, _ = step_pair
    result = change_two_values(start)
    got = used(result, "CO2", {AREA: "COL"})
    # no old value: the source of the step; with old value: computed from it
    assert got[2001] == {"const"}
    assert got[2002] == {"CO2/COL/2002"}
    assert got[2000] == {"CO2/COL/2000"}
    assert used(result, "CO2", {AREA: "MEX"}) == used(start, "CO2", {AREA: "MEX"})


def test_decorator_replaces():
    start = tracked_per_point(
        make_ds(
            make_da(
                "CO2",
                [[1, NAN, 3, NAN], [NAN, 2, NAN, 4]],
                {AREA: ["COL", "MEX"], "time": years(2000, 2003)},
            )
        )
    )

    @primap2.records_processing_step(
        "ds", function="change", description="changed", source="const", replaces=True
    )
    def change(ds):
        return change_two_values.__wrapped__(ds)

    got = used(change(start), "CO2", {AREA: "COL"})
    assert got[2001] == {"const"}
    assert got[2002] == {"const"}
    assert got[2000] == {"CO2/COL/2000"}


def test_decorator_other(step_pair):
    start, other = step_pair
    got = used(take_from_other(start, other), "CO2", {AREA: "COL"})
    assert got[2000] == {"CO2/COL/2000"}
    assert got[2001] == {"o:CO2/COL/2001"}
    # overwritten: computed from the old value and the value of other
    assert got[2002] == {"CO2/COL/2002", "o:CO2/COL/2002"}


def test_decorator_other_replaces(step_pair):
    start, other = step_pair
    got = used(take_from_other_replacing(start, other), "CO2", {AREA: "COL"})
    assert got[2000] == {"CO2/COL/2000"}
    assert got[2001] == {"o:CO2/COL/2001"}
    assert got[2002] == {"o:CO2/COL/2002"}


def test_processing_step_default(step_pair):
    start, _ = step_pair
    with start.pr.processing_step(function="change", description="changed", source="const") as step:
        step.ds = change_two_values.__wrapped__(step.ds)
    got = used(step.result, "CO2", {AREA: "COL"})
    assert got[2001] == {"const"}
    assert got[2002] == {"CO2/COL/2002"}


def test_processing_step_replaces(step_pair):
    start, _ = step_pair
    with start.pr.processing_step(
        function="change", description="changed", source="const", replaces=True
    ) as step:
        step.ds = change_two_values.__wrapped__(step.ds)
    got = used(step.result, "CO2", {AREA: "COL"})
    assert got[2001] == {"const"}
    assert got[2002] == {"const"}
    assert got[2000] == {"CO2/COL/2000"}


def test_processing_step_other(step_pair):
    start, other = step_pair
    with start.pr.processing_step(function="take", description="taken", other_ds=other) as step:
        step.ds["CO2"].loc[{AREA: "COL", "time": ["2001", "2002"]}] = other["CO2"].loc[
            {AREA: "COL", "time": ["2001", "2002"]}
        ]
    got = used(step.result, "CO2", {AREA: "COL"})
    assert got[2001] == {"o:CO2/COL/2001"}
    assert got[2002] == {"CO2/COL/2002", "o:CO2/COL/2002"}
    with start.pr.processing_step(
        function="take", description="taken", other_ds=other, replaces=True
    ) as step:
        step.ds["CO2"].loc[{AREA: "COL", "time": ["2001", "2002"]}] = other["CO2"].loc[
            {AREA: "COL", "time": ["2001", "2002"]}
        ]
    got = used(step.result, "CO2", {AREA: "COL"})
    assert got[2001] == {"o:CO2/COL/2001"}
    assert got[2002] == {"o:CO2/COL/2002"}


def test_compose_local_trends_boundaries_and_gap():
    n = 20
    a = [float(i + 1) for i in range(n)]
    for i in (0, 1, 2, 8, 9, 10, 18, 19):
        a[i] = NAN
    b = [2.0 * (i + 1) + 1 for i in range(n)]
    ds = csg_input({"A": a, "B": b})
    fit_params = primap2.csg.FitParameters(
        fit_degree=1, trend_length=3, min_trend_points=3, fallback_degree=0
    )
    result = compose(
        ds,
        [
            ({SRC: "A"}, primap2.csg.SubstitutionStrategy()),
            ({}, primap2.csg.LocalTrendsStrategy(fit_params=fit_params)),
        ],
    )

    def both(*ys):
        return {f"CH4/{src}/COL/{y}" for src in "AB" for y in ys}

    got = used(result, "CH4", {AREA: "COL"})
    assert set(got) == set(range(2000, 2020))
    for y in (2000, 2001, 2002):
        assert got[y] == {f"CH4/B/COL/{y}"} | both(2003, 2004, 2005)
    for y in (2008, 2009, 2010):
        assert got[y] == {f"CH4/B/COL/{y}"} | both(2005, 2006, 2007, 2011, 2012, 2013)
    for y in (2018, 2019):
        assert got[y] == {f"CH4/B/COL/{y}"} | both(2015, 2016, 2017)


@pytest.mark.parametrize("allow_shift", [True, False])
def test_compose_global_least_squares_shift_variants(allow_shift):
    ds = csg_input({"A": [1, 2, NAN, NAN, 5], "B": [2, 4, 6, 8, NAN]})
    result = compose(
        ds,
        [
            ({SRC: "A"}, primap2.csg.SubstitutionStrategy()),
            ({}, primap2.csg.GlobalLSStrategy(allow_shift=allow_shift)),
        ],
    )
    fit = {f"CH4/{s}/COL/{y}" for s in "AB" for y in (2000, 2001)}
    got = used(result, "CH4", {AREA: "COL"})
    assert got == {
        2000: {"CH4/A/COL/2000"},
        2001: {"CH4/A/COL/2001"},
        2002: {"CH4/B/COL/2002"} | fit,
        2003: {"CH4/B/COL/2003"} | fit,
        2004: {"CH4/A/COL/2004"},
    }


def test_compose_falls_back_to_substitution_without_overlap():
    ds = csg_input({"A": [1, 2, NAN, NAN], "B": [NAN, NAN, 3, 4]})
    result = compose(
        ds,
        [({}, primap2.csg.GlobalLSStrategy()), ({}, primap2.csg.SubstitutionStrategy())],
    )
    assert used(result, "CH4", {AREA: "COL"}) == {
        2000: {"CH4/A/COL/2000"},
        2001: {"CH4/A/COL/2001"},
        2002: {"CH4/B/COL/2002"},
        2003: {"CH4/B/COL/2003"},
    }


def test_compose_local_trends_windows_with_gaps_next_to_gaps():
    """The windows of ts hold only the data points which exist, fill_ts has more data
    in the same windows."""
    n = 14
    a = [float(i + 1) for i in range(n)]
    for i in (3, 4, 6, 7, 8):
        a[i] = NAN
    b = [2.0 * (i + 1) + 1 for i in range(n)]
    ds = csg_input({"A": a, "B": b})
    fit_params = primap2.csg.FitParameters(
        fit_degree=1, trend_length=3, min_trend_points=3, fallback_degree=0
    )
    result = compose(
        ds,
        [
            ({SRC: "A"}, primap2.csg.SubstitutionStrategy()),
            ({}, primap2.csg.LocalTrendsStrategy(fit_params=fit_params)),
        ],
    )

    def s(src, *ys):
        return {f"CH4/{src}/COL/{y}" for y in ys}

    got = used(result, "CH4", {AREA: "COL"})
    assert set(got) == set(range(2000, 2014))
    # gap 2003-2004: windows 2000-2002 and 2005-2007, of which only 2005 has data in A
    for y in (2003, 2004):
        assert got[y] == s("B", y) | s("A", 2000, 2001, 2002, 2005) | s(
            "B", 2000, 2001, 2002, 2005, 2006, 2007
        )
    # gap 2006-2008: windows 2003-2005, of which only 2005 has data in A, and 2009-2011
    for y in (2006, 2007, 2008):
        assert got[y] == s("B", y) | s("A", 2005, 2009, 2010, 2011) | s(
            "B", 2003, 2004, 2005, 2009, 2010, 2011
        )
    assert got[2005] == s("A", 2005)
