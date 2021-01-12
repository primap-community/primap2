import logging

import pytest
from loguru import logger


# see https://loguru.readthedocs.io/en/stable/resources/migration.html#making-things-work-with-pytest-and-caplog  # noqa: E501
@pytest.fixture
def caplog(_caplog):
    class PropogateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropogateHandler(), format="{message} {extra}")
    yield _caplog
    logger.remove(handler_id)
