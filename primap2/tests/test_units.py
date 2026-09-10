"""Tests for _units.py"""

import logging

import numpy as np
import pytest
import xarray as xr
import xarray.testing

from primap2 import ureg

from .utils import allclose, assert_equal


def warnings(caplog) -> list[logging.LogRecord]:
    """The records logged at WARNING level or above."""
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_roundtrip_quantify(opulent_ds: xr.Dataset):
    roundtrip = opulent_ds.pr.dequantify().pr.quantify()
    xarray.testing.assert_identical(roundtrip, opulent_ds)


def test_roundtrip_quantify_da(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6 (SARGWP100)"]
    roundtrip = da.pr.dequantify().pr.quantify()
    assert_equal(roundtrip, da)


def test_convert_to_gwp(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6"]
    da_converted = da.pr.convert_to_gwp("SARGWP100", "CO2 Gg / year")
    da_expected = opulent_ds["SF6 (SARGWP100)"]
    assert_equal(da_converted, da_expected)

    da_converted_like = da.pr.convert_to_gwp_like(da_expected)
    assert_equal(da_converted_like, da_expected)


def test_convert_to_gwp_like_missing(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6"]
    da_gwp = da.pr.convert_to_gwp("SARGWP100", "CO2 Gg / year")

    del da_gwp.attrs["gwp_context"]
    with pytest.raises(ValueError, match="reference array has no gwp_context"):
        da.pr.convert_to_gwp_like(da_gwp)

    da_gwp = xr.full_like(da_gwp, np.nan)
    da_gwp.attrs["gwp_context"] = "SARGWP100"
    with pytest.raises(ValueError, match="reference array has no units attached"):
        da.pr.convert_to_gwp_like(da_gwp)


def test_convert_to_gwp_other_context(opulent_ds: xr.Dataset):
    """A single gas in another metric is converted back to mass automatically."""
    da: xr.DataArray = opulent_ds["SF6 (SARGWP100)"]
    da_converted = da.pr.convert_to_gwp("AR4GWP100", "CO2 Gg / year")

    da_expected = opulent_ds["SF6"].pr.convert_to_gwp("AR4GWP100", "CO2 Gg / year")
    assert_equal(da_converted, da_expected)
    # the input is not modified by the detour via the mass
    assert da.attrs["gwp_context"] == "SARGWP100"


def test_convert_to_gwp_incompatible(empty_ds: xr.Dataset):
    """A gas basket has no mass, so it can not be converted to another metric."""
    da: xr.DataArray = empty_ds["KYOTOGHG (AR4GWP100)"]
    with pytest.raises(ValueError, match="Incompatible GWP conversions"):
        da.pr.convert_to_gwp("AR6GWP100", "CO2 Gg / year")


def test_convert_to_mass(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6 (SARGWP100)"]
    da_converted = da.pr.convert_to_mass()
    da_expected = opulent_ds["SF6"]
    assert_equal(da_converted, da_expected)


def test_convert_round_trip(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6"]
    assert da.attrs["entity"] == "SF6"
    da_gwp = da.pr.convert_to_gwp(gwp_context="AR4GWP100", units="Gg CO2 / year")
    da_rt = da_gwp.pr.convert_to_mass()
    assert_equal(da, da_rt)
    assert da_rt.attrs["entity"] == "SF6"
    assert isinstance(da_rt.attrs["entity"], str)


def test_convert_to_mass_missing_info(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6"]
    with pytest.raises(
        ValueError,
        match="No gwp_context given and no gwp_context available in the attrs",
    ):
        da.pr.convert_to_mass()

    da = opulent_ds["SF6 (SARGWP100)"]
    del da.attrs["entity"]
    with pytest.raises(ValueError, match="No entity given and no entity available in the attrs"):
        da.pr.convert_to_mass()


class TestDatasetConvertToGWP:
    @pytest.fixture
    def gases_ds(self, opulent_processing_ds: xr.Dataset) -> xr.Dataset:
        """Opulent dataset without the variable which is already a GWP.

        Converting ``SF6 (SARGWP100)`` collides with the conversion of ``SF6``, which
        is tested separately.
        """
        return opulent_processing_ds.drop_vars(["SF6 (SARGWP100)", "Processing of SF6 (SARGWP100)"])

    def test_convert(self, gases_ds: xr.Dataset, caplog):
        converted = gases_ds.pr.convert_to_gwp("AR4GWP100", "Gg CO2 / year")

        # gases are converted and renamed, everything else is kept unchanged
        assert set(converted.data_vars) == {
            "CO2 (AR4GWP100)",
            "SF6 (AR4GWP100)",
            "CH4 (AR4GWP100)",
            "population",
            "Processing of CO2 (AR4GWP100)",
            "Processing of SF6 (AR4GWP100)",
            "Processing of CH4 (AR4GWP100)",
            "Processing of population",
        }
        converted.pr.ensure_valid()
        assert "Not converting" in caplog.text
        assert "population" in caplog.text

        assert_equal(
            converted["SF6 (AR4GWP100)"],
            gases_ds["SF6"].pr.convert_to_gwp("AR4GWP100", "Gg CO2 / year"),
        )
        assert_equal(converted["population"], gases_ds["population"])

    def test_processing_info_renamed(self, gases_ds: xr.Dataset):
        converted = gases_ds.pr.convert_to_gwp("AR4GWP100", "Gg CO2 / year")

        processing = converted["Processing of CO2 (AR4GWP100)"]
        assert processing.attrs["described_variable"] == "CO2 (AR4GWP100)"
        assert processing.attrs["entity"] == "Processing of CO2 (AR4GWP100)"
        # processing info of variables which were not converted is left alone
        assert converted["Processing of population"].attrs["described_variable"] == "population"

    def test_gas_basket_same_context(self, empty_ds: xr.Dataset):
        """A gas basket can be converted within its own context, that is a unit change."""
        converted = empty_ds.pr.convert_to_gwp("AR4GWP100", "Mt CO2 / year")

        assert "KYOTOGHG (AR4GWP100)" in converted
        converted.pr.ensure_valid()
        assert_equal(
            converted["KYOTOGHG (AR4GWP100)"],
            empty_ds["KYOTOGHG (AR4GWP100)"].pint.to("Mt CO2 / year"),
        )
        # assert_equal converts the units before comparing, so check them explicitly:
        # all variables of the result have to be given in the requested units
        for variable in converted.data_vars:
            assert converted[variable].pint.units == ureg.Unit("Mt CO2 / year")

    def test_other_context_converted(self, opulent_ds: xr.Dataset):
        """A single gas in another metric is converted back to mass automatically."""
        ds = opulent_ds.drop_vars(["SF6"])
        converted = ds.pr.convert_to_gwp("AR4GWP100", "Gg CO2 / year")

        assert "SF6 (AR4GWP100)" in converted
        converted.pr.ensure_valid()
        assert_equal(
            converted["SF6 (AR4GWP100)"],
            opulent_ds["SF6"].pr.convert_to_gwp("AR4GWP100", "Gg CO2 / year"),
        )

    def test_gas_basket_other_context_warns(self, empty_ds: xr.Dataset, caplog):
        """A gas basket in another metric can not be converted at all, so it is kept.

        This is not an error, but as the dataset does not contain the gas basket in
        the requested metric either, the result mixes global warming potentials.
        """
        converted = empty_ds.pr.convert_to_gwp("AR6GWP100", "Gg CO2 / year")

        assert_equal(converted["KYOTOGHG (AR4GWP100)"], empty_ds["KYOTOGHG (AR4GWP100)"])
        assert "CO2 (AR6GWP100)" in converted
        converted.pr.ensure_valid()

        assert warnings(caplog)
        assert "mixes global warming potentials" in caplog.text
        assert "KYOTOGHG (AR4GWP100)" in caplog.text

    def test_realistic(self, realistic_ds: xr.Dataset, caplog):
        """Real data has gases as masses and gas baskets in several metrics."""
        converted = realistic_ds.pr.convert_to_gwp("AR6GWP100", "Gg CO2 / year")

        assert set(converted.data_vars) == {
            "CO2 (AR6GWP100)",
            "CH4 (AR6GWP100)",
            "N2O (AR6GWP100)",
            "SF6 (AR6GWP100)",
            "HFCS (AR5GWP100)",
            "HFCS (AR6GWP100)",
        }
        converted.pr.ensure_valid()

        # HFCS is also given in the requested metric, so keeping HFCS (AR5GWP100)
        # loses nothing and must not warn
        assert_equal(converted["HFCS (AR5GWP100)"], realistic_ds["HFCS (AR5GWP100)"])
        assert not warnings(caplog)
        assert "nothing is missing from the result" in caplog.text

    def test_realistic_context_not_available_warns(self, realistic_ds: xr.Dataset, caplog):
        """If the gas baskets are in no metric we asked for, the result is mixed."""
        converted = realistic_ds.pr.convert_to_gwp("AR4GWP100", "Gg CO2 / year")

        assert_equal(converted["HFCS (AR5GWP100)"], realistic_ds["HFCS (AR5GWP100)"])
        assert_equal(converted["HFCS (AR6GWP100)"], realistic_ds["HFCS (AR6GWP100)"])
        assert warnings(caplog)
        assert "mixes global warming potentials" in caplog.text

    def test_name_collision_raises(self, opulent_ds: xr.Dataset):
        """Converting SF6 and SF6 (SARGWP100) both give SF6 (AR4GWP100)."""
        with pytest.raises(
            ValueError,
            match=r"Converting 'SF6 \(SARGWP100\)' would overwrite 'SF6 \(AR4GWP100\)'",
        ):
            opulent_ds.pr.convert_to_gwp("AR4GWP100", "Gg CO2 / year")

    def test_convert_like(self, opulent_ds: xr.Dataset):
        ds = opulent_ds.drop_vars(["SF6"])
        converted = ds.pr.convert_to_gwp_like(opulent_ds["SF6 (SARGWP100)"])

        assert "CH4 (SARGWP100)" in converted
        converted.pr.ensure_valid()
        assert_equal(
            converted["CH4 (SARGWP100)"],
            opulent_ds["CH4"].pr.convert_to_gwp_like(opulent_ds["SF6 (SARGWP100)"]),
        )

    def test_convert_like_missing(self, opulent_ds: xr.Dataset):
        like = opulent_ds["SF6 (SARGWP100)"].copy()
        del like.attrs["gwp_context"]
        with pytest.raises(ValueError, match="reference array has no gwp_context"):
            opulent_ds.pr.convert_to_gwp_like(like)

        like = xr.full_like(opulent_ds["SF6 (SARGWP100)"], np.nan)
        like.attrs["gwp_context"] = "SARGWP100"
        with pytest.raises(ValueError, match="reference array has no units attached"):
            opulent_ds.pr.convert_to_gwp_like(like)


class TestDatasetConvertToMass:
    def test_convert(self, empty_ds: xr.Dataset, caplog):
        ds = empty_ds.pr.convert_to_gwp("AR4GWP100", "Gg CO2 / year")
        caplog.clear()
        converted = ds.pr.convert_to_mass()

        # the gases are converted back, the gas basket has no mass and is kept
        assert set(converted.data_vars) == {"CO2", "SF6", "CH4", "KYOTOGHG (AR4GWP100)"}
        converted.pr.ensure_valid()

        # this is the expected shape of a published dataset, so it must not warn
        assert not warnings(caplog)
        assert "KYOTOGHG (AR4GWP100)" in caplog.text

    def test_realistic(self, realistic_ds: xr.Dataset, caplog):
        """Gases are already masses, the gas baskets can not be converted.

        The dataset is already in the shape convert_to_mass produces, so it stays
        exactly as it is and nothing is warned about.
        """
        converted = realistic_ds.pr.convert_to_mass()

        xarray.testing.assert_identical(converted, realistic_ds)
        assert not warnings(caplog)
        assert "HFCS (AR5GWP100)" in caplog.text
        assert "HFCS (AR6GWP100)" in caplog.text

    def test_round_trip(self, minimal_ds: xr.Dataset):
        ds = minimal_ds.drop_vars(["SF6 (SARGWP100)"])
        round_trip = ds.pr.convert_to_gwp("AR4GWP100", "Gg CO2 / year").pr.convert_to_mass()

        assert set(round_trip.data_vars) == set(ds.data_vars)
        for variable in ds.data_vars:
            assert allclose(round_trip[variable], ds[variable])

    def test_processing_info_renamed(self, opulent_processing_ds: xr.Dataset):
        ds = opulent_processing_ds.drop_vars(["SF6", "Processing of SF6"])
        converted = ds.pr.convert_to_mass()

        assert "Processing of SF6" in converted
        assert converted["Processing of SF6"].attrs["described_variable"] == "SF6"
        assert converted["Processing of SF6"].attrs["entity"] == "Processing of SF6"
        converted.pr.ensure_valid()

    def test_not_a_gwp_kept(self, minimal_ds: xr.Dataset, caplog):
        ds = minimal_ds.drop_vars(["SF6 (SARGWP100)"])
        converted = ds.pr.convert_to_mass()

        # nothing is a global warming potential, so the dataset is unchanged
        xarray.testing.assert_identical(converted, ds)
        assert "Not converting" in caplog.text

    def test_name_collision_raises(self, minimal_ds: xr.Dataset):
        """SF6 (SARGWP100) converts to SF6, which already exists."""
        with pytest.raises(
            ValueError, match=r"Converting 'SF6 \(SARGWP100\)' would overwrite 'SF6'"
        ):
            minimal_ds.pr.convert_to_mass()


def test_context(opulent_ds: xr.Dataset):
    da: xr.DataArray = opulent_ds["SF6 (SARGWP100)"]
    with da.pr.gwp_context:
        da_converted = opulent_ds["SF6"].pint.to(da.pint.units)
    assert allclose(da, da_converted)
