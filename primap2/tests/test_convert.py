"""Tests for _convert.py"""

import climate_categories as cc
import climate_categories._conversions as conversions
import pytest
import xarray as xr
import pathlib
import primap2
import numpy as np


def test_convert_ipcc(empty_ds: xr.Dataset):
    # build a DA categorized by IPCC1996 and with 1 everywhere so results are easy
    # to see
    da = empty_ds["CO2"]
    da = da.expand_dims({"category (IPCC1996)": list(cc.IPCC1996.keys())})
    da = da.expand_dims({"source (gas)": list(cc.gas.keys())})
    arr = da.data.copy()
    arr[:] = 1 * primap2.ureg("Gg CO2 / year")
    da.data = arr

    with pytest.raises(ValueError, match="The conversion uses auxiliary categories"):
        da.pr.convert("category", "IPCC2006", sum_rule="extensive")

    result = da.pr.convert(
        "category",
        "IPCC2006",
        sum_rule="extensive",
        auxiliary_dimensions={"gas": "source (gas)"},
    )

    assert (result.pr.loc[{"category": "1"}] == 1.0 * primap2.ureg("Gg CO2 / year")).all().item()
    assert (result.pr.loc[{"category": "2"}] == 2.0 * primap2.ureg("Gg CO2 / year")).all().item()

def test_convert_BURDI(empty_ds: xr.Dataset):
    # make a sample conversion object in climate categories
    filepath = pathlib.Path("data/BURDI_conversion.csv")
    conv = conversions.ConversionSpec.from_csv(filepath)
    conv = conv.hydrate(cats=cc.cats["BURDI"]._cats)

    # taken from UNFCCC_non-AnnexI_data/src/unfccc_ghg_data/unfccc_di_reader/
    # unfccc_di_reader_config.py
    BURDI_categories = [
            "1",
            "1.A",
            "1.A.1",
            "1.A.2",
            "1.A.3",
            "1.A.4",
            "1.A.5",
            "1.B",
            "1.B.1",
            "1.B.2",
            "2",
            "2.A",
            "2.B",
            "2.C",
            "2.D",
            "2.E",
            "2.F",
            "2.G",
            "3",
            "4",
            "4.A",
            "4.B",
            "4.C",
            "4.D",
            "4.E",
            "4.F",
            "4.G",
            "5",
            "6",
            "6.A",
            "6.B",
            "6.C",
            "6.D",
            "24540",
            "15163",
            "14637",
            "14424",
            "14423",
            "14638",
            "7"]


    # build a DA categorized by BURDI and with 1 everywhere so results are easy
    # to see
    da = empty_ds["CO2"]
    da = da.expand_dims({"category (BURDI)": BURDI_categories})
    da = da.expand_dims({"source (gas)": list(cc.gas.keys())})
    arr = da.data.copy()
    arr[:] = 1 * primap2.ureg("Gg CO2 / year")
    da.data = arr

    result = da.pr.convert(
        "category",
        conv,
        sum_rule="extensive",
        auxiliary_dimensions={"gas" : "source (gas)"},
    )

    # cat 2 + 3 in BURDI equals cat 2 in IPCC2006_PRIMAP
    assert (result.pr.loc[{"category": "2"}] == 2.0 * primap2.ureg("Gg CO2 / year")).all().item()
    # cat 4.D + 4.C + 4.E + 4.F + 4.G in BURDI equals cat 3.C in IPCC2006_PRIMAP
    assert (result.pr.loc[{"category" : "3.C"}] == 5.0 * primap2.ureg("Gg CO2 / year")).all().item()
    # cat 5 in BURDI equals cat M.LULUCF in IPCC2006_PRIMAP
    assert (result.pr.loc[{"category" : "M.LULUCF"}] == 1.0 * primap2.ureg("Gg CO2 / year")).all().item()
    # 2.E + 2.B = 2.E, 2.E should not be part of new data set
    assert np.isnan(result.pr.loc[{"category" : "2.E"}].values).all()
    # cat 14638 in BURDI equals cat M.BIO in IPCC2006_PRIMAP
    # TODO: This will fail. M.BIO is currently not listed in climate categories
    assert (result.pr.loc[{"category" : "M.BIO"}] == 1.0 * primap2.ureg("Gg CO2 / year")).all().item()