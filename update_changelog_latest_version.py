"""Create or update the .changelog_latest_version.md file.

It contains the entries from the changelog, but just for the latest version. It is used
for describing individual releases.
"""

import pathlib


def main():
    with (
        pathlib.Path(".changelog_latest_version.md").open("w") as changelog_latest_version,
        pathlib.Path("changelog.md").open("r") as changelog,
    ):
        clv_content = changelog.read().split("##")[1]
        clv_lines = clv_content.split("\n")
        version = clv_lines[0].strip()
        clv_changes = "\n".join(clv_lines[1:]).strip()
        changelog_latest_version.write(f"""# primap2 release {version}

primap2 is a library for compiling and analyzing climate policy datasets.

Important changes in this release:

{clv_changes}
""")


if __name__ == "__main__":
    main()
