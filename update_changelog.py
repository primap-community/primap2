#!/usr/bin/env python3
import pathlib
import sys

version = sys.argv[1]

with open("CHANGELOG.rst") as fd:
    old_content = fd.read().splitlines(keepends=True)

with open("CHANGELOG.rst", "w") as fd:
    # Write header
    fd.writelines(old_content[:4])

    # New version
    fd.write(f"{version}\n")
    fd.write("-" * len(version) + "\n")
    ch_dir = pathlib.Path("changelog_unreleased")
    for changelog_file in ch_dir.iterdir():
        fd.write(changelog_file.read_text())
        changelog_file.unlink()
    fd.write("\n")

    # Rest
    fd.writelines(old_content[4:])
