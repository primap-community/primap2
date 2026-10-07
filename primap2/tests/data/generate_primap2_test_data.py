"""Regenerate the csg regression test input ``primap2_test_data_v2.5.1_final.nc``.

The fixture is a small subset of the per-source input data of PRIMAP-hist v2.5.1_final,
used by ``primap2/tests/csg/test_wrapper.py::test_create_composite_source``.

The input data is not part of this repository, it lives in the (separate) PRIMAP-hist
data repository under ``v2.5.1_final/input/``. Pass the path to that directory:

.. code-block:: shell

    python primap2/tests/data/generate_primap2_test_data.py ~/PRIMAP-hist_data/v2.5.1_final/input

Note that regenerating the input data invalidates the expected result in
``PRIMAP-csg-test.nc``, which has to be regenerated as well if the values change.
"""

import argparse
import pathlib

import xarray as xr

import primap2

#: the per-source input datasets of PRIMAP-hist which are sampled
INPUT_DATASETS = (
    "ANDREW2023V4I",
    "CDIAC2023I",
    "CRF2023I",
    "UNFCCCALLI",
    "EI2023I",
    "FAO2024AIPMH",
    "EDGAR80I",
    "EDGAR70I",
    "HOUGHTONDWNI",
)
#: a handful of categories, countries, and entities to keep the fixture small, while
#: still covering gas baskets, multiple global warming potentials, and categories which
#: are only reported by some of the sources
CATEGORIES = ("1.A", "1.B.2", "2.A", "2.B", "2", "M.AG.ELV", "4", "M.LULUCF")
COUNTRIES = ("DEU", "USA", "BRA", "TUV", "HRV", "CHN", "GHA")
ENTITIES = ("CO2", "CH4", "N2O", "HFCS (AR6GWP100)", "HFCS (AR5GWP100)", "SF6")
TIME = slice("1960", "2022")

COMPRESSION = {"zlib": True, "complevel": 9}


def stack_source_scen(ds: xr.Dataset, scenario_terminology: str = "PRIMAP") -> xr.Dataset:
    """Combine the source and scenario dimensions into a single source dimension.

    The individual input datasets use a scenario dimension to distinguish e.g. the
    different releases of a source, while the csg uses ``source`` as its only priority
    dimension. Therefore, both are combined into source values of the form
    ``"{source}, {scenario}"``.
    """
    stacked = ds.stack(
        dimensions={"SourceScen": ["source", f"scenario ({scenario_terminology})"]},
        create_index=True,
    )
    source_scen = [f"{source}, {scenario}" for source, scenario in stacked["SourceScen"].to_numpy()]
    stacked = stacked.assign_coords({"SourceScenario": ("SourceScen", source_scen)})
    stacked = stacked.drop_vars(["source", f"scenario ({scenario_terminology})", "SourceScen"])
    stacked = stacked.set_index(SourceScen="SourceScenario", append=True)
    # remove empty source-scenario combinations
    return stacked.dropna("SourceScen", how="all")


def generate(input_path: pathlib.Path) -> xr.Dataset:
    """Read the per-source input datasets and combine a subset of them into one dataset."""
    combined = None
    for source in INPUT_DATASETS:
        ds = primap2.open_dataset(input_path / f"{source}.nc")
        ds = ds[[entity for entity in ds.data_vars if entity in ENTITIES]]
        ds = ds.pr.loc[
            {
                "time": TIME,
                "category": [
                    cat for cat in ds["category (IPCC2006_PRIMAP)"].to_numpy() if cat in CATEGORIES
                ],
                "area": [
                    country for country in ds["area (ISO3)"].to_numpy() if country in COUNTRIES
                ],
            }
        ]
        ds = stack_source_scen(ds)
        combined = ds if combined is None else combined.pr.merge(ds)

    combined = combined.rename({"SourceScen": "source"})
    # the scenario dimension was folded into the source dimension by stack_source_scen,
    # so the attribute pointing at it has to go as well
    del combined.attrs["scen"]
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=pathlib.Path,
        help="the 'input' directory of PRIMAP-hist v2.5.1_final",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "primap2_test_data_v2.5.1_final.nc",
        help="where to write the generated dataset",
    )
    args = parser.parse_args()

    ds = generate(args.input_path)
    ds.pr.ensure_valid()
    ds.pr.to_netcdf(args.output, encoding={var: COMPRESSION for var in ds.data_vars})


if __name__ == "__main__":
    main()
