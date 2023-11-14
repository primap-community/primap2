"""Check if the used version of Python is good enough for us."""
# ruff: noqa
import sys

MIN_VERSION = (3, 9)

if sys.version_info < (3, 9):
    min_version_human_readable = ".".join(str(x) for x in MIN_VERSION)
    print(
        f"Python version {sys.version_info} not supported, please install Python"
        f" version {min_version_human_readable} or higher"
    )
    sys.exit(1)
