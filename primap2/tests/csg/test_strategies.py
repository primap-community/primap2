import numpy as np

import primap2.csg
from primap2.tests.csg.utils import get_single_ts


def test_substitution_strategy():
    ts = get_single_ts(data=1.0)
    ts[0] = np.nan
    fill_ts = get_single_ts(data=2.0)

    (result_ts, result_descriptions) = primap2.csg.SubstitutionStrategy().fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    assert result_ts[0] == 2.0
    assert (result_ts[1:] == 1.0).all()
    assert len(result_descriptions) == 1
    assert result_descriptions[0].time == np.array(["1850"], dtype=np.datetime64)
    assert (
        result_descriptions[0].description
        == "substituted with corresponding values from B"
    )
    assert "source" not in result_ts.coords.keys()


def test_null_strategy():
    ts = get_single_ts(data=1.0)
    fill_ts = get_single_ts(data=2.0)

    result_ts, result_desc = primap2.csg.NullStrategy().fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="R"
    )
    assert result_ts.isnull().all()
    assert len(result_desc) == 1
    assert result_desc[0].time == "all"
    assert result_desc[0].description == "filled with NaN values, not using data from R"
