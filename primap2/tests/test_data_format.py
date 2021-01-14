#!/usr/bin/env python
"""Tests for the ``primap2`` package - data format tests."""
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pint_xarray
import pytest
import xarray as xr
from openscm_units import unit_registry as ureg

import primap2

pint_xarray.accessors.setup_registry(ureg)


@pytest.fixture
def minimal_ds():
    """A valid, minimal dataset."""
    time = pd.date_range("2000-01-01", "2020-01-01", freq="AS")
    area_iso3 = np.array(["COL", "ARG", "MEX", "BOL"])
    minimal = xr.Dataset(
        {
            ent: xr.DataArray(
                data=np.random.rand(len(time), len(area_iso3), 1),
                coords={
                    "time": time,
                    "area (ISO3)": area_iso3,
                    "source": ["RAND2020"],
                },
                dims=["time", "area (ISO3)", "source"],
                attrs={"units": f"{ent} Gg / year", "entity": ent},
            )
            for ent in ("CO2", "SF6", "CH4")
        },
        attrs={"area": "area (ISO3)"},
    ).pint.quantify(unit_registry=ureg)

    with ureg.context("SARGWP100"):
        minimal["SF6 (SARGWP100)"] = minimal["SF6"].pint.to("CO2 Gg / year")
    minimal["SF6 (SARGWP100)"].attrs["gwp_context"] = "SARGWP100"
    return minimal


@pytest.fixture
def opulent_ds():
    """A valid dataset using lots of features."""
    coords = {
        "time": pd.date_range("2000-01-01", "2020-01-01", freq="AS"),
        "area (ISO3)": np.array(["COL", "ARG", "MEX", "BOL"]),
        "category (IPCC 2006)": np.array(["0", "1", "2", "3", "4", "5", "1.A", "1.B"]),
        "animal (FAOSTAT)": np.array(["cow", "swine", "goat"]),
        "scenario (FAOSTAT)": np.array(["highpop", "lowpop"]),
        "provenance": np.array(["projected"]),
        "model": np.array(["FANCYFAO"]),
        "source": np.array(["RAND2020"]),
    }

    opulent = xr.Dataset(
        {
            ent: xr.DataArray(
                data=np.random.rand(*(len(x) for x in coords.values())),
                coords=coords,
                dims=list(coords.keys()),
                attrs={"units": f"{ent} Gg / year", "entity": ent},
            )
            for ent in ("CO2", "SF6", "CH4")
        },
        attrs={
            "area": "area (ISO3)",
            "cat": "category (IPCC 2006)",
            "sec_cats": [
                "animal (FAOSTAT)",
            ],
            "scen": "scenario (FAOSTAT)",
            "reference": "doi:10.1012",
            "rights": "Use however you want.",
            "contact": "lol_no_one_will_answer@example.com",
            "description": "GHG inventory data ...",
        },
    )

    pop_coords = {
        x: coords[x]
        for x in (
            "time",
            "area (ISO3)",
            "provenance",
            "model",
            "source",
        )
    }
    opulent["population"] = xr.DataArray(
        data=np.random.rand(*(len(x) for x in pop_coords.values())),
        coords=pop_coords,
        dims=list(pop_coords.keys()),
        attrs={"entity": "population", "units": ""},
    )

    opulent = opulent.assign_coords(
        {
            "category_names": xr.DataArray(
                data=np.array(
                    [
                        "total",
                        "industry",
                        "energy",
                        "transportation",
                        "residential",
                        "land use",
                        "heavy industry",
                        "light industry",
                    ]
                ),
                coords={"category (IPCC 2006)": coords["category (IPCC 2006)"]},
                dims=["category (IPCC 2006)"],
            )
        }
    )

    opulent = opulent.pint.quantify(unit_registry=ureg)

    with ureg.context("SARGWP100"):
        opulent["SF6 (SARGWP100)"] = opulent["SF6"].pint.to("CO2 Gg / year")
    opulent["SF6 (SARGWP100)"].attrs["gwp_context"] = "SARGWP100"

    return opulent


def test_valid_ds_pass(minimal_ds, opulent_ds, caplog):
    """Valid datasets should pass inspection."""
    primap2.ensure_valid(minimal_ds)
    primap2.ensure_valid(opulent_ds)
    assert not caplog.records


def test_io_roundtrip(minimal_ds, opulent_ds, caplog):
    with tempfile.TemporaryDirectory() as tempdir:
        tpath = pathlib.Path(tempdir)
        primap2.save(minimal_ds, tpath / "minimal.nc")
        primap2.save(opulent_ds, tpath / "opulent.nc")
        primap2.ensure_valid(primap2.load(tpath / "minimal.nc"))
        primap2.ensure_valid(primap2.load(tpath / "opulent.nc"))
    assert not caplog.records
