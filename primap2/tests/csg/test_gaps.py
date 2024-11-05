import numpy as np
import pandas as pd
import pytest
import xarray as xr

from primap2.csg._strategies.gaps import get_gaps


@pytest.fixture
def test_ts() -> xr.DataArray:
    ts = xr.DataArray(
        np.concatenate(
            [
                np.array([np.nan] * 4),
                np.linspace(6, 12, 11),
                np.array([np.nan] * 6),
                np.linspace(15, 4, 4),
                np.array([np.nan] * 14),
                np.linspace(5, 3, 15),
                np.array([np.nan] * 20),
            ]
        ),
        coords={"time": pd.date_range("1952-01-01", "2025-01-01", freq="YS")},
        dims="time",
        name="test_ts",
    )
    return ts


def test_get_gaps_left_boundary(test_ts):
    gaps = get_gaps(test_ts.pr.loc[{"time": slice("1952", "1960")}])
    assert len(gaps) == 1
    assert gaps[0].left == np.datetime64("1952-01-01")
    assert gaps[0].right == np.datetime64("1955-01-01")
    assert gaps[0].type == "l"


def test_get_gaps_right_boundary(test_ts):
    gaps = get_gaps(test_ts.pr.loc[{"time": slice("2000", "2025")}])
    assert len(gaps) == 1
    assert gaps[0].left == np.datetime64("2006-01-01")
    assert gaps[0].right == np.datetime64("2025-01-01")
    assert gaps[0].type == "r"


def test_get_gaps_gaps(test_ts):
    gaps = get_gaps(test_ts.pr.loc[{"time": slice("1960", "2002")}])
    assert len(gaps) == 2
    assert gaps[0].left == np.datetime64("1967-01-01")
    assert gaps[0].right == np.datetime64("1972-01-01")
    assert gaps[0].type == "g"
    assert gaps[1].left == np.datetime64("1977-01-01")
    assert gaps[1].right == np.datetime64("1990-01-01")
    assert gaps[1].type == "g"


def test_get_gaps_full(test_ts):
    gaps = get_gaps(test_ts)
    assert len(gaps) == 4
    assert gaps[0].left == np.datetime64("1952-01-01")
    assert gaps[0].right == np.datetime64("1955-01-01")
    assert gaps[0].type == "l"
    assert gaps[1].left == np.datetime64("2006-01-01")
    assert gaps[1].right == np.datetime64("2025-01-01")
    assert gaps[1].type == "r"
    assert gaps[2].left == np.datetime64("1967-01-01")
    assert gaps[2].right == np.datetime64("1972-01-01")
    assert gaps[2].type == "g"
    assert gaps[3].left == np.datetime64("1977-01-01")
    assert gaps[3].right == np.datetime64("1990-01-01")
    assert gaps[3].type == "g"
