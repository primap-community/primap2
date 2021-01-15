#!/usr/bin/env python
"""Tests for the ``primap2`` package - data format tests."""
import logging

import numpy as np
import pandas as pd
import pint
import pytest
import xarray as xr

import primap2
from primap2 import ureg


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
    ).pr.quantify()

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
        "product (FAOSTAT)": np.array(["milk", "meat"]),
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
            "sec_cats": ["animal (FAOSTAT)", "product (FAOSTAT)"],
            "scen": "scenario (FAOSTAT)",
            "references": "doi:10.1012",
            "rights": "Use however you want.",
            "contact": "lol_no_one_will_answer@example.com",
            "title": "Completely invented GHG inventory data",
            "comment": "GHG inventory data ...",
            "institution": "PIK",
            "history": """2021-01-14 14:50 data invented
                2021-01-14 14:51 additional processing step
                """,
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


def test_something_else_entirely(caplog):
    with pytest.raises(ValueError, match=r"ds is not an xr.Dataset"):
        something_else = primap2._data_format.DatasetDataFormatAccessor()
        something_else._ds = "asdf"
        something_else.ensure_valid()
    assert "ERROR" in caplog.text
    assert "object is not an xarray Dataset." in caplog.text


def test_valid_ds_pass(minimal_ds, opulent_ds, caplog):
    caplog.set_level(logging.INFO)
    minimal_ds.pr.ensure_valid()
    opulent_ds.pr.ensure_valid()
    assert not caplog.records


def test_io_roundtrip(minimal_ds, opulent_ds, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    minimal_ds.pr.to_netcdf(tmp_path / "minimal.nc")
    opulent_ds.pr.to_netcdf(tmp_path / "opulent.nc")
    primap2.open_dataset(tmp_path / "minimal.nc").pr.ensure_valid()
    primap2.open_dataset(tmp_path / "opulent.nc").pr.ensure_valid()
    assert not caplog.records


def test_required_dimension_missing(caplog):
    ds = xr.Dataset(
        {
            "area (ISO3)": ["a"],
            "time": pd.date_range("2000-01-01", "2020-01-01", freq="AS"),
        },
        attrs={"area": "area (ISO3)"},
    ).pr.quantify()

    with pytest.raises(ValueError, match=r"'source' not in dims"):
        ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "'source' not found in dims, but is required." in caplog.text


def test_required_coordinate_missing(minimal_ds, caplog):
    del minimal_ds["source"]
    with pytest.raises(ValueError, match=r"dim 'source' has no coord"):
        minimal_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "No coord found for dimension 'source'." in caplog.text


def test_dimension_metadata_missing(minimal_ds, caplog):
    del minimal_ds.attrs["area"]
    with pytest.raises(ValueError, match=r"'area' not in attrs"):
        minimal_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert (
        "'area' not found in attrs, required dimension is therefore undefined."
        in caplog.text
    )


def test_dimension_metadata_wrong(minimal_ds, caplog):
    minimal_ds.attrs["area"] = "asdf"
    with pytest.raises(ValueError, match=r"'area' dimension not in dims"):
        minimal_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "'asdf' defined as 'area' dimension, but not found in dims." in caplog.text


def test_wrong_provenance_value(opulent_ds, caplog):
    opulent_ds["provenance"] = ["asdf"]
    with pytest.raises(ValueError, match=r"Invalid provenance: \{'asdf'\}"):
        opulent_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "provenance contains invalid values: {'asdf'}" in caplog.text


def test_additional_dimension(minimal_ds: xr.Dataset, caplog):
    ds = minimal_ds.expand_dims({"addl_dim": ["a", "b", "c"]})
    ds.pr.ensure_valid()
    assert "WARNING" in caplog.text
    assert (
        "Dimension(s) {'addl_dim'} unknown, likely a typo or missing in sec_cats."
        in caplog.text
    )


def test_wrong_dimension_key(minimal_ds, caplog):
    ds = minimal_ds.rename_dims({"area (ISO3)": "asdf"})
    ds.attrs["area"] = "asdf"
    with pytest.raises(
        ValueError, match=r"'asdf' not in the format 'dim \(category_set\)'"
    ):
        ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "'asdf' not in the format 'dim (category_set)'." in caplog.text


def test_not_one_source(minimal_ds: xr.Dataset, caplog):
    ds = minimal_ds.loc[{"source": "RAND2020"}]
    ds["source"] = []
    with pytest.raises(ValueError, match=r"Exactly one source required"):
        ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "Exactly one source required per data set." in caplog.text

    caplog.clear()
    ds["source"] = ["a", "b"]
    with pytest.raises(ValueError, match=r"Exactly one source required"):
        ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "Exactly one source required per data set." in caplog.text


def test_missing_sec_cat(minimal_ds, caplog):
    minimal_ds.attrs["sec_cats"] = ["missing"]
    with pytest.raises(ValueError, match="Secondary category 'missing' not in dims"):
        minimal_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "Secondary category 'missing' defined, but not found in dims." in caplog.text


def test_missing_optional_dim(minimal_ds, caplog):
    minimal_ds.attrs["scen"] = "missing"
    with pytest.raises(ValueError, match="'scen' not in dims"):
        minimal_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "'missing' defined as scen, but not found in dims." in caplog.text


def test_sec_cat_without_primary_cat(minimal_ds, caplog):
    ds = minimal_ds.expand_dims({"something (cset)": ["a", "b", "c"]})
    ds.attrs["sec_cats"] = ["something (cset)"]
    ds.pr.ensure_valid()
    assert "WARNING" in caplog.text
    assert (
        "Secondary category defined, but no primary category defined, weird."
        in caplog.text
    )


def test_additional_coordinate_space(opulent_ds: xr.Dataset, caplog):
    ds = opulent_ds.rename({"category_names": "category names"})
    with pytest.raises(
        ValueError, match=r"Coord key 'category names' contains a space"
    ):
        ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert (
        "Additional coordinate name 'category names' contains a space,"
        " replace it with an underscore." in caplog.text
    )


def test_missing_entity(minimal_ds, caplog):
    del minimal_ds["CO2"].attrs["entity"]
    with pytest.raises(ValueError, match="entity missing for 'CO2'"):
        minimal_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "'CO2' has no entity declared in attributes." in caplog.text


def test_weird_entity(minimal_ds, caplog):
    minimal_ds["CO2"].attrs["entity"] = "carbondioxide"
    minimal_ds.pr.ensure_valid()
    assert "WARNING" in caplog.text
    assert "entity 'carbondioxide' of 'CO2' is unknown." in caplog.text


def test_missing_gwp_context(minimal_ds, caplog):
    del minimal_ds["SF6 (SARGWP100)"].attrs["gwp_context"]
    minimal_ds.pr.ensure_valid()
    assert "WARNING" in caplog.text
    assert (
        "'SF6 (SARGWP100)' has the dimension [CO2 * mass / time], but is not CO2. "
        "gwp_context missing?" in caplog.text
    )


def test_wrong_units(minimal_ds, caplog):
    deq = minimal_ds.pint.dequantify()
    deq["CO2"].attrs["units"] = "kg CO2"
    req = deq.pr.quantify()
    req.pr.ensure_valid()
    assert "WARNING" in caplog.text
    assert (
        "'CO2' has a unit of CO2 * kilogram, which is not compatible with an emission "
        "rate." in caplog.text
    )


def test_unquantified(minimal_ds, caplog):
    deq = minimal_ds.pint.dequantify()
    deq.pr.ensure_valid()
    assert not caplog.records


def test_multi_units(minimal_ds, caplog):
    minimal_ds["CO2"].attrs["units"] = "kg CO2 / year"
    with pytest.raises(ValueError, match="data already has units"):
        minimal_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "'units' in variable attrs, but data is quantified already." in caplog.text


def test_invalid_units(minimal_ds, caplog):
    deq = minimal_ds.pint.dequantify()
    deq["CO2"].attrs["units"] = "i_am_not_a_unit"
    with pytest.raises(pint.UndefinedUnitError):
        deq.pr.ensure_valid()


def test_invalid_gwp_context(minimal_ds, caplog):
    minimal_ds["SF6 (SARGWP100)"].attrs["gwp_context"] = "i_am_not_a_gwp_context"
    with pytest.raises(
        ValueError,
        match=r"Invalid gwp_context 'i_am_not_a_gwp_context' for 'SF6 \(SARGWP100\)'",
    ):
        minimal_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert (
        "gwp_context 'i_am_not_a_gwp_context' for 'SF6 (SARGWP100)' is not valid."
        in caplog.text
    )


def test_extraneous_gwp_context(minimal_ds, caplog):
    minimal_ds["SF6"].attrs["gwp_context"] = "SARGWP100"
    with pytest.raises(
        ValueError, match=r"SF6 has wrong dimensionality for gwp_context."
    ):
        minimal_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert (
        "'SF6' is a global warming potential, but the dimension is not "
        "[CO2 * mass / time]." in caplog.text
    )


def test_missing_unit(minimal_ds, caplog):
    minimal_ds["CO2"] = (minimal_ds["CO2"].dims, minimal_ds["CO2"].pint.magnitude)
    minimal_ds["CO2"].attrs["entity"] = "CO2"
    with pytest.raises(ValueError, match=r"units missing for 'CO2'"):
        minimal_ds.pr.ensure_valid()
    assert "ERROR" in caplog.text
    assert "'CO2' has no units." in caplog.text


def test_weird_variable_name(minimal_ds, caplog):
    caplog.set_level(logging.INFO)
    minimal_ds["weird_name"] = minimal_ds["CO2"]
    minimal_ds["weird_name"].attrs["entity"] = "CO2"
    minimal_ds.pr.ensure_valid()
    assert "INFO" in caplog.text
    assert "The name 'weird_name' is not in standard format 'CO2'." in caplog.text


def test_missing_gwp_in_variable_name(minimal_ds, caplog):
    minimal_ds["SF6_gwp"] = minimal_ds["SF6 (SARGWP100)"]
    minimal_ds.pr.ensure_valid()
    assert "WARNING" in caplog.text
    assert "'SF6_gwp' has a gwp_context in attrs, but not in its name." in caplog.text


def test_weird_contact(minimal_ds, caplog):
    caplog.set_level(logging.INFO)
    minimal_ds.attrs["contact"] = "i_am_not_an_email"
    minimal_ds.pr.ensure_valid()
    assert "INFO" in caplog.text
    assert "Contact information is not an email address" in caplog.text


def test_weird_references(minimal_ds, caplog):
    caplog.set_level(logging.INFO)
    minimal_ds.attrs["references"] = "i_am_not_a_doi"
    minimal_ds.pr.ensure_valid()
    assert "INFO" in caplog.text
    assert "Reference information is not a DOI" in caplog.text


def test_nonstandard_attribute(minimal_ds, caplog):
    minimal_ds.attrs["i_am_not_standard"] = ""
    minimal_ds.pr.ensure_valid()
    assert "WARNING" in caplog.text
    assert "Unknown metadata in attrs: {'i_am_not_standard'}, typo?" in caplog.text
