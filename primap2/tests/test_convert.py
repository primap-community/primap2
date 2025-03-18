"""Tests for _convert.py"""

import importlib
import importlib.resources
import re

import climate_categories as cc
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2


def get_test_data_filepath(fname: str):
    return importlib.resources.files("primap2.tests.data").joinpath(fname)


@pytest.mark.xfail
def test_conversion_source_does_not_match_dataset_dimension(empty_ds):
    # make a data set with IPCC1996 categories
    da = empty_ds["CO2"]
    da = da.expand_dims({"category (IPCC1996)": list(cc.IPCC1996.keys())})
    da = da.expand_dims({"source (gas)": list(cc.gas.keys())})
    arr = da.data.copy()
    arr[:] = 1 * primap2.ureg("Gg CO2 / year")
    da.data = arr

    # load the BURDI to IPCC2006 category conversion
    filepath = get_test_data_filepath("BURDI_conversion.csv")
    conv = cc.Conversion.from_csv(filepath)

    msg = (
        "The source categorization in the conversion (BURDI) "
        "does not match the categorization in the data set (IPCC1996)."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        result = da.pr.convert(  # noqa: F841
            dim="category",
            conversion=conv,
        )


def test_convert_ipcc(empty_ds: xr.Dataset):
    # build a DA categorized by IPCC1996 and with 1 everywhere so results are easy
    # to see
    da = empty_ds["CO2"]
    da = da.expand_dims({"category (IPCC1996)": list(cc.IPCC1996.keys())})
    da = da.expand_dims({"source (gas)": list(cc.gas.keys())})
    arr = da.data.copy()
    arr[:] = 1 * primap2.ureg("Gg CO2 / year")
    da.data = arr

    conversion = cc.IPCC1996.conversion_to(cc.IPCC2006)

    with pytest.raises(ValueError, match="The conversion uses auxiliary categories"):
        da.pr.convert(
            dim="category",
            conversion=conversion,
        )

    result = da.pr.convert(
        dim="category",
        conversion=conversion,
        auxiliary_dimensions={"gas": "source (gas)"},
    )
    # rule 1 -> 1
    assert (result.pr.loc[{"category": "1"}] == 1.0 * primap2.ureg("Gg CO2 / year")).all().item()
    # rule 2 + 3 -> 2
    assert (result.pr.loc[{"category": "2"}] == 2.0 * primap2.ureg("Gg CO2 / year")).all().item()
    # rule 1.A.2.f -> 1.A.2.f + 1.A.2.g + 1.A.2.h + 1.A.2.i + 1.A.2.j + 1.A.2.k + 1.A.2.l + 1.A.2.m
    autocat = "A_(1.A.2.f+1.A.2.g+1.A.2.h+1.A.2.i+1.A.2.j+1.A.2.k+1.A.2.l+1.A.2.m)"
    assert (
        (result.pr.loc[{"category": autocat}] == 8.0 * primap2.ureg("Gg CO2 / year")).all().item()
    )
    # rule 4.D for N2O only -> 3.C.4 + 3.C.5
    autocat = "A_(3.C.4+3.C.5)"
    assert (
        (
            result.pr.loc[{"category": autocat, "source (gas)": "N2O"}]
            == 2.0 * primap2.ureg("Gg CO2 / year")
        )
        .all()
        .item()
    )
    # all other gases should be nan
    all_gases_but_N2O = list(result.indexes["source (gas)"])
    all_gases_but_N2O.remove("N2O")
    assert np.isnan(
        result.pr.loc[{"category": autocat, "source (gas)": all_gases_but_N2O}].values
    ).all()
    # rule 7 -> 5
    assert (result.pr.loc[{"category": "5"}] == 1.0 * primap2.ureg("Gg CO2 / year")).all().item()
    # rule 2.F.6 -> 2.E + 2.F.6 + 2.G.1 + 2.G.2 + 2.G.4,
    # rule 2.F.6 + 3.D -> 2.E + 2.F.6 + 2.G - ignored because 2.F.G already converted
    # rule 2.G -> 2.H.3 - 1-to-1-conversion
    autocat = "A_(2.E+2.F.6+2.G.1+2.G.2+2.G.4)"
    assert (
        (result.pr.loc[{"category": autocat}] == 5.0 * primap2.ureg("Gg CO2 / year")).all().item()
    )
    assert "A_(2.E+2.F.6+2.G)" not in list(result.indexes["category (IPCC2006)"])
    assert (
        (result.pr.loc[{"category": "2.H.3"}] == 1.0 * primap2.ureg("Gg CO2 / year")).all().item()
    )


# test with new conversion and two existing categorisations
@pytest.mark.xfail
def test_convert_BURDI(empty_ds: xr.Dataset):
    # make a sample conversion object in climate categories
    filepath = get_test_data_filepath("BURDI_conversion.csv")
    conv = cc.Conversion.from_csv(filepath)

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
        "7",
    ]

    # build a DA categorised by BURDI and with 1 everywhere so results are easy
    # to see
    da = empty_ds["CO2"]
    da = da.expand_dims({"category (BURDI)": BURDI_categories})
    da = da.expand_dims({"source (gas)": list(cc.gas.keys())})
    arr = da.data.copy()
    arr[:] = 1 * primap2.ureg("Gg CO2 / year")
    da.data = arr

    result = da.pr.convert(
        dim="category",
        conversion=conv,
        auxiliary_dimensions={"gas": "source (gas)"},
    )

    # cat 2 + 3 in BURDI equals cat 2 in IPCC2006_PRIMAP
    assert (result.pr.loc[{"category": "2"}] == 2.0 * primap2.ureg("Gg CO2 / year")).all().item()
    # cat 4.D + 4.C + 4.E + 4.F + 4.G in BURDI equals cat 3.C in IPCC2006_PRIMAP
    assert (result.pr.loc[{"category": "3.C"}] == 5.0 * primap2.ureg("Gg CO2 / year")).all().item()
    # cat 5 in BURDI equals cat M.LULUCF in IPCC2006_PRIMAP
    assert (
        (result.pr.loc[{"category": "M.LULUCF"}] == 1.0 * primap2.ureg("Gg CO2 / year"))
        .all()
        .item()
    )
    # 3.C.7 (converted from 4.C) should still be part of the data set,
    # although it appears in two conversion rules
    assert (
        (result.pr.loc[{"category": "3.C.7"}] == 1.0 * primap2.ureg("Gg CO2 / year")).all().item()
    )
    # rule 2.E + 2.B -> 2.B
    # 2.E is part of PRIMAP categories, but cannot be retrieved from conversion
    assert np.isnan(result.pr.loc[{"category": "2.E"}].values).all()
    # cat 14638 in BURDI equals cat M.BIO in IPCC2006_PRIMAP
    assert (
        (result.pr.loc[{"category": "M.BIO"}] == 1.0 * primap2.ureg("Gg CO2 / year")).all().item()
    )
    # 4.D -> M.3.C.45.AG
    # TODO This category is only available on M3C45AG branch in climate categories
    # test locally with:
    # `source venv/bin/activate`
    # `pip install -e ../climate_categories`
    # Will pass after climate categories release
    assert (
        (result.pr.loc[{"category": "M.3.C.45.AG"}] == 1.0 * primap2.ureg("Gg CO2 / year"))
        .all()
        .item()
    )


# test with new conversion and new categorisations
def test_custom_conversion_and_two_custom_categorisations(empty_ds):
    # make categorisation A from yaml
    categorisation_a = cc.from_yaml(get_test_data_filepath("simple_categorisation_a.yaml"))

    # make categorisation B from yaml
    categorisation_b = cc.from_yaml(get_test_data_filepath("simple_categorisation_b.yaml"))

    # categories not part of climate categories so we need to add them manually
    cats = {
        "A": categorisation_a,
        "B": categorisation_b,
    }

    # make conversion from csv
    conv = cc.Conversion.from_csv(get_test_data_filepath("simple_conversion.csv"), cats=cats)

    # make a dummy dataset based on A cats
    da = empty_ds["CO2"]
    da = da.expand_dims({"category (A)": list(categorisation_a.keys())})
    arr = da.data.copy()
    arr[:] = 1 * primap2.ureg("Gg CO2 / year")
    da.data = arr

    # convert to categorisation B
    result = da.pr.convert(
        dim="category",
        conversion=conv,
    )

    # category name includes B - the target categorisation
    assert sorted(result.coords) == ["area (ISO3)", "category (B)", "source", "time"]

    # check 1 -> 1
    assert (result.pr.loc[{"category": "1"}] == 1.0 * primap2.ureg("Gg CO2 / year")).all().item()

    # check 2 + 3 -> 2
    assert (result.pr.loc[{"category": "2"}] == 2.0 * primap2.ureg("Gg CO2 / year")).all().item()

    # check result has 2 categories (input categorisation had 3)
    # TODO this is ambiguous, order may change
    assert result.shape == (5, 21, 4, 1)


def test_nan_conversion(empty_ds):
    # make categorisation A from yaml
    categorisation_a = cc.from_yaml(get_test_data_filepath("simple_categorisation_a.yaml"))

    # make categorisation B from yaml
    categorisation_b = cc.from_yaml(get_test_data_filepath("simple_categorisation_b.yaml"))

    # categories not part of climate categories so we need to add them manually
    cats = {
        "A": categorisation_a,
        "B": categorisation_b,
    }

    # make conversion from csv
    conv = cc.Conversion.from_csv(get_test_data_filepath("simple_conversion.csv"), cats=cats)

    # make a dummy dataset based on A cats
    da = empty_ds["CO2"]
    da = da.expand_dims({"category (A)": list(categorisation_a.keys())})
    arr = da.data.copy()
    arr[:] = 1 * primap2.ureg("Gg CO2 / year")
    da.data = arr
    # set some values to nan
    da.loc[{"category (A)": "1", "area (ISO3)": "MEX"}] = np.nan * primap2.ureg("Gg CO2 / year")

    # convert to categorisation B
    result = da.pr.convert(
        dim="category",
        conversion=conv,
    )

    # check that the nan value is still nan
    assert all(np.isnan(result.loc[{"category (B)": "1", "area (ISO3)": "MEX"}].to_numpy()))


def test_create_category_name():
    # make categorisation A from yaml
    categorisation_a = cc.from_yaml(get_test_data_filepath("simple_categorisation_a.yaml"))

    # make categorisation B from yaml
    categorisation_b = cc.from_yaml(get_test_data_filepath("simple_categorisation_b.yaml"))

    # categories not part of climate categories so we need to add them manually
    cats = {
        "A": categorisation_a,
        "B": categorisation_b,
    }

    # make conversion from csv
    conv = cc.Conversion.from_csv(
        get_test_data_filepath("test_create_category_name_conversion.csv"), cats=cats
    )

    # check that first positive category does not have '+' sign
    autocat = primap2._convert.create_category_name(conv.rules[0])
    assert autocat == "A_(1+2)"

    # check that first negative category has '-' sign
    autocat = primap2._convert.create_category_name(conv.rules[1])
    assert autocat == "A_(-3+4)"

    autocat = primap2._convert.create_category_name(conv.rules[2])
    assert autocat == "A_(5-1)"


def convert_dataframe(df, conv):
    # Not implemented in this example:
    # * deal with auxiliary dimensions
    # * deal with factors

    # make an empty dataframe for new data
    df_converted = pd.DataFrame(columns=df.columns)

    one_to_one_rules = []
    one_to_n_rules = []
    for rule in conv.rules:
        # We would also need to check the factor is one
        if rule.cardinality_a == "one" and rule.cardinality_b == "one":
            one_to_one_rules.append(rule)
        else:
            one_to_n_rules.append(rule)

    # fill one to one rules first (if there are any)
    one_to_one_slices = []
    for rule in one_to_one_rules:
        category = next([cat.codes[0] for cat in rule.factors_categories_a.keys()])
        df_filtered = df.loc[df["category (A)"] == category]
        one_to_one_slices.append(df_filtered)

    # we can simply add the rows for one to one conversion to the data frame
    df_converted = pd.concat(one_to_one_slices, join="outer")

    # now add all the one to n rules
    one_to_n_slices = []
    for rule in one_to_n_rules:
        categories = [cat.codes[0] for cat in rule.factors_categories_a.keys()]
        # filter all by the categories on the left side of the rule
        df_filtered = df.loc[df["category (A)"].isin(categories)]
        # group by all columns but the years and sum the values
        df_filtered = (
            df_filtered.groupby(["area (ISO3)", "source", "entity", "unit"]).sum().reset_index()
        )
        # assign new category value
        df_filtered["category (A)"] = next(
            [cat.codes[0] for cat in rule.factors_categories_b.keys()]
        )
        one_to_n_slices.append(df_filtered)

    df_converted = pd.concat([df_converted, *one_to_n_slices], join="outer")

    df_converted = df_converted.rename(columns={"category (A)": "category (B)"})

    return df_converted


def test_conversion_with_dataframes(empty_ds):
    # make categorisation A from yaml
    categorisation_a = cc.from_yaml(get_test_data_filepath("simple_categorisation_a.yaml"))

    # make categorisation B from yaml
    categorisation_b = cc.from_yaml(get_test_data_filepath("simple_categorisation_b.yaml"))

    # categories not part of climate categories so we need to add them manually
    cats = {
        "A": categorisation_a,
        "B": categorisation_b,
    }

    # make conversion from csv
    conv = cc.Conversion.from_csv(get_test_data_filepath("simple_conversion.csv"), cats=cats)

    # make a dummy dataset based on A cats
    da = empty_ds["CO2"]
    da = da.expand_dims({"category (A)": list(categorisation_a.keys())})
    arr = da.data.copy()
    arr[:] = 1 * primap2.ureg("Gg CO2 / year")
    da.data = arr

    ds = xr.Dataset({"CO2": da})

    df = ds.pr.to_interchange_format()

    result = convert_dataframe(df, conv)

    years = list(np.datetime_as_string(empty_ds.time.to_numpy(), unit="Y"))

    # check 1 -> 1
    assert (result.loc[result["category (B)"] == "1", years].to_numpy() == 1).all()

    # check 2 + 3 -> 2
    assert (result.loc[result["category (B)"] == "2", years].to_numpy() == 2).all()
