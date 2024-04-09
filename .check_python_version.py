"""Check if the used version of Python is good enough for us."""

import itertools
import sys

SUPPORTED_MAJOR_VERSIONS = (3,)
SUPPORTED_MINOR_VERSIONS = (10, 11)

if (
    sys.version_info.major not in SUPPORTED_MAJOR_VERSIONS
    or sys.version_info.minor not in SUPPORTED_MINOR_VERSIONS
):
    supported_versions = itertools.product(
        SUPPORTED_MAJOR_VERSIONS, SUPPORTED_MINOR_VERSIONS
    )
    supported_versions_human_readable = ", ".join(
        ".".join(str(x) for x in version) for version in supported_versions
    )
    print(
        f"Python version {sys.version_info} not supported, please install Python"
        f" in one of the supported versions: {supported_versions_human_readable}."
    )
    sys.exit(1)
