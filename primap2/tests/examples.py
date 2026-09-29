import datetime

import numpy as np
import pandas as pd
import xarray as xr

import primap2
from primap2 import ureg
from primap2._processing_info import processing_variable_name


def minimal_ds() -> xr.Dataset:
    """A valid, minimal dataset."""
    time = pd.date_range("2000-01-01", "2020-01-01", freq="YS")
    area_iso3 = np.array(["COL", "ARG", "MEX", "BOL"])

    # seed the rng with a constant to achieve predictable "randomness"
    rng = np.random.default_rng(1)

    minimal = xr.Dataset(
        {
            ent: xr.DataArray(
                data=rng.random((len(time), len(area_iso3), 1)),
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

    return minimal


def minimal_ds_in_gwp() -> xr.Dataset:
    """Like the minimal dataset, but with all gases given as a global warming potential."""
    return minimal_ds().pr.convert_to_gwp(gwp_context="SARGWP100", units="CO2 Gg / year")


def toy_ds() -> xr.Dataset:
    """A toy dataset which can be used to demonstrate a lot of concepts."""
    time = pd.date_range("2015-01-01", "2020-01-01", freq="YS")
    area_iso3 = np.array(["COL", "ARG"])
    cat = np.array(["0", "1", "2", "1.A", "1.B"])

    # seed the rng with a constant to achieve predictable "randomness"
    rng = np.random.default_rng(1)

    toy = xr.Dataset(
        {
            ent: xr.DataArray(
                data=rng.random((len(time), len(area_iso3), len(cat), 2)),
                coords={
                    "time": time,
                    "area (ISO3)": area_iso3,
                    "category (IPCC2006)": cat,
                    "source": ["RAND2020", "RAND2021"],
                },
                dims=["time", "area (ISO3)", "category (IPCC2006)", "source"],
                attrs={"units": f"{ent} Gg / year", "entity": ent},
            )
            for ent in ("CO2", "CH4")
        },
        attrs={"area": "area (ISO3)", "cat": "category (IPCC2006)"},
    ).pr.quantify()

    return toy


COORDS = {
    "time": pd.date_range("2000-01-01", "2020-01-01", freq="YS"),
    "area (ISO3)": np.array(["COL", "ARG", "MEX", "BOL"]),
    "category (IPCC 2006)": np.array(["0", "1", "2", "3", "4", "5", "1.A", "1.B"]),
    "animal (FAOSTAT)": np.array(["cow", "swine", "goat"]),
    "product (FAOSTAT)": np.array(["milk", "meat"]),
    "scenario (FAOSTAT)": np.array(["highpop", "lowpop"]),
    "provenance": np.array(["projected"]),
    "model": np.array(["FANCYFAO"]),
    "source": np.array(["RAND2020", "RAND2021"]),
}


def opulent_ds() -> xr.Dataset:
    """A valid dataset using lots of features."""
    # seed the rng with a constant to achieve predictable "randomness"
    rng = np.random.default_rng(1)

    opulent = xr.Dataset(
        {
            ent: xr.DataArray(
                data=rng.random(tuple(len(x) for x in COORDS.values())),
                coords=COORDS,
                dims=list(COORDS.keys()),
                attrs={"units": f"{ent} Gg / year", "entity": ent},
            )
            for ent in ("CO2", "SF6", "CH4")
        },
        attrs={
            "entity_terminology": "primap2",
            "area": "area (ISO3)",
            "cat": "category (IPCC 2006)",
            "scen": "scenario (FAOSTAT)",
            "references": "doi:10.1012",
            "rights": "Use however you want.",
            "contact": "lol_no_one_will_answer@example.com",
            "title": "Completely invented GHG inventory data",
            "comment": "GHG inventory data ...",
            "institution": "PIK",
            "publication_date": datetime.date(2099, 12, 31),
        },
    )

    pop_coords = {
        x: COORDS[x]
        for x in (
            "time",
            "area (ISO3)",
            "provenance",
            "model",
            "source",
        )
    }
    pop_shape = tuple(len(x) for x in pop_coords.values())
    opulent["population"] = xr.DataArray(
        data=rng.random(pop_shape),
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
                coords={"category (IPCC 2006)": COORDS["category (IPCC 2006)"]},
                dims=["category (IPCC 2006)"],
            )
        }
    )

    opulent = opulent.pint.quantify(unit_registry=ureg)

    return opulent


def opulent_str_ds() -> xr.Dataset:
    """Like the opulent dataset, but additionally with a stringly typed data variable
    "method".
    """
    opulent = opulent_ds()

    method_coords = {
        x: COORDS[x]
        for x in (
            "time",
            "area (ISO3)",
            "model",
            "source",
        )
    }
    method_shape = tuple(len(x) for x in method_coords.values())
    opulent["method"] = xr.DataArray(
        data=np.ones(method_shape, dtype=str).astype(object),
        coords=method_coords,
        dims=list(method_coords.keys()),
        attrs={"entity": "method"},
    )
    opulent["method"].pr.loc[{"time": "2000", "area": "COL", "source": "RAND2020"}] = "text"

    return opulent


def opulent_processing_ds() -> xr.Dataset:
    """Like the opulent dataset, but additionally with processing information data
    variables.
    """
    opulent = opulent_ds()

    new_vars = {}
    for var in opulent:
        # processing information has the same dimensions as the variable it describes,
        # with the exception of "time"
        dims = [dim for dim in opulent[var].dims if dim != "time"]
        shape = tuple(len(opulent[x]) for x in dims)
        new_vars[processing_variable_name(var)] = xr.DataArray(
            data=np.full(
                shape=shape,
                fill_value=primap2.TimeseriesProcessingDescription(
                    steps=[
                        primap2.ProcessingStepDescription(
                            time="all",
                            function="random",
                            description="Values created randomly.",
                        )
                    ]
                ),
            ),
            coords=opulent[dims],
            dims=dims,
            attrs={
                "entity": processing_variable_name(var),
                "described_variable": var,
            },
        )

    opulent.update(new_vars)

    return opulent


def empty_ds() -> xr.Dataset:
    """An empty hull of a dataset with missing data."""
    time = pd.date_range("2000-01-01", "2020-01-01", freq="YS")
    area_iso3 = np.array(["COL", "ARG", "MEX", "BOL"])
    coords = {
        "time": time,
        "area (ISO3)": area_iso3,
        "source": ["RAND2020"],
    }
    dims = ["time", "area (ISO3)", "source"]
    empty = xr.Dataset(
        {
            ent: xr.DataArray(
                data=np.zeros((len(time), len(area_iso3), 1), dtype=float),
                coords=coords,
                dims=dims,
                attrs={"units": f"{ent} Gg / year", "entity": ent},
            )
            for ent in ("CO2", "SF6", "CH4")
        },
        attrs={"area": "area (ISO3)"},
    ).pr.quantify()

    empty["KYOTOGHG (AR4GWP100)"] = xr.DataArray(
        data=np.zeros((len(time), len(area_iso3), 1), dtype=float),
        coords=coords,
        dims=dims,
        attrs={
            "units": "CO2 Gg / year",
            "entity": "KYOTOGHG",
            "gwp_context": "AR4GWP100",
        },
    ).pr.quantify()

    return empty


def realistic_ds() -> xr.Dataset:
    """A dataset shaped like the data published by PRIMAP.

    In contrast to the other examples, single gases are only given as masses and
    global warming potentials are only given for gas baskets, with the same gas
    basket given in more than one global warming potential metric. No gas is
    present both as a mass and as a global warming potential.
    """
    time = pd.date_range("2000-01-01", "2020-01-01", freq="YS")
    area_iso3 = np.array(["COL", "ARG", "MEX", "BOL"])
    coords = {
        "time": time,
        "area (ISO3)": area_iso3,
        "source": ["RAND2020"],
    }
    dims = ["time", "area (ISO3)", "source"]
    shape = (len(time), len(area_iso3), 1)

    # seed the rng with a constant to achieve predictable "randomness"
    rng = np.random.default_rng(1)

    realistic = xr.Dataset(
        {
            ent: xr.DataArray(
                data=rng.random(shape),
                coords=coords,
                dims=dims,
                attrs={"units": f"{ent} Gg / year", "entity": ent},
            )
            for ent in ("CO2", "CH4", "N2O", "SF6")
        },
        attrs={"area": "area (ISO3)"},
    ).pr.quantify()

    for gwp_context in ("AR5GWP100", "AR6GWP100"):
        realistic[f"HFCS ({gwp_context})"] = xr.DataArray(
            data=rng.random(shape),
            coords=coords,
            dims=dims,
            attrs={
                "units": "CO2 Gg / year",
                "entity": "HFCS",
                "gwp_context": gwp_context,
            },
        ).pr.quantify()

    return realistic


_cached_minimal_ds = minimal_ds()
_cached_minimal_ds_in_gwp = minimal_ds_in_gwp()
_cached_opulent_ds = opulent_ds()
_cached_opulent_str_ds = opulent_str_ds()
_cached_opulent_processing_ds = opulent_processing_ds()
_cached_empty_ds = empty_ds()
_cached_realistic_ds = realistic_ds()
