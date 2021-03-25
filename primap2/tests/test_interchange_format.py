"""Tests for the interchange format."""

import xarray as xr

from primap2 import pm2io

from . import utils


def test_round_trip(any_ds: xr.Dataset, tmp_path):
    path = tmp_path / "if"
    pm2io.write_interchange_format(path, any_ds.pr.to_interchange_format())
    with path.with_suffix(".yaml").open() as fd:
        print(fd.read())
    actual = pm2io.from_interchange_format(pm2io.read_interchange_format(path))
    utils.assert_ds_aligned_equal(any_ds, actual)
