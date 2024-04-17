import logging

import pytest
import xarray as xr
from loguru import logger

import primap2  # noqa: F401

from . import examples


# monkey-patch caplog to work wit loguru
# see https://loguru.readthedocs.io/en/stable/resources/migration.html#making-things-work-with-pytest-and-caplog
@pytest.fixture
def caplog(caplog):
    class PropogateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropogateHandler(), format="{message} {extra}")
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def minimal_ds() -> xr.Dataset:
    """A valid, minimal dataset."""
    return examples._cached_minimal_ds.copy(deep=True)


@pytest.fixture
def opulent_ds() -> xr.Dataset:
    """A valid dataset using lots of features."""
    return examples._cached_opulent_ds.copy(deep=True)


@pytest.fixture
def opulent_str_ds() -> xr.Dataset:
    """Like the opulent dataset, but additionally with a stringly typed data variable
    "method"."""
    return examples._cached_opulent_str_ds.copy(deep=True)


@pytest.fixture
def empty_ds() -> xr.Dataset:
    """An empty hull of a dataset with missing data."""
    return examples._cached_empty_ds.copy(deep=True)


@pytest.fixture
def opulent_processing_ds() -> xr.Dataset:
    """Like the opulent dataset, but additionally with processing information."""
    return examples._cached_opulent_processing_ds.copy(deep=True)


@pytest.fixture(
    params=["opulent", "opulent_str", "opulent_processing", "minimal", "empty"]
)
def any_ds(request) -> xr.Dataset:
    """Test with all available valid example Datasets."""
    if request.param == "opulent":
        return examples._cached_opulent_ds.copy(deep=True)
    elif request.param == "opulent_str":
        return examples._cached_opulent_str_ds.copy(deep=True)
    elif request.param == "opulent_processing":
        return examples._cached_opulent_processing_ds.copy(deep=True)
    elif request.param == "minimal":
        return examples._cached_minimal_ds.copy(deep=True)
    elif request.param == "empty":
        return examples._cached_empty_ds.copy(deep=True)
