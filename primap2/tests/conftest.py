import logging

import pytest
from loguru import logger

import primap2  # noqa: F401

from . import examples


# monkey-patch caplog to work wit loguru
# see https://loguru.readthedocs.io/en/stable/resources/migration.html#making-things-work-with-pytest-and-caplog  # noqa: E501
@pytest.fixture
def caplog(caplog):
    class PropogateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropogateHandler(), format="{message} {extra}")
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def minimal_ds():
    """A valid, minimal dataset."""
    return examples.minimal_ds()


@pytest.fixture
def opulent_ds():
    """A valid dataset using lots of features."""
    return examples.opulent_ds()


@pytest.fixture
def empty_ds():
    """An empty hull of a dataset with missing data."""
    return examples.empty_ds()


@pytest.fixture(params=["opulent", "minimal", "empty"])
def any_ds(request):
    """Test with all available valid example Datasets."""
    if request.param == "opulent":
        return examples.opulent_ds()
    elif request.param == "minimal":
        return examples.minimal_ds()
    elif request.param == "empty":
        return examples.empty_ds()
    else:
        raise ValueError
