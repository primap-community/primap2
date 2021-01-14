import logging

import pytest
from loguru import logger


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
