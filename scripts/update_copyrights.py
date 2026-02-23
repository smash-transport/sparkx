#!/usr/bin/env python3
# ===================================================
#
#    Copyright (c) 2026
#      SPARKX Team
#
#    GNU General Public License (GPLv3 or later)
#
# ===================================================
"""
Update copyright year ranges in SPARKX .py files based on git history.

For each .py file in src/sparkx/ (recursively) and tests/, this script:
1. Queries `git log` for all commit years that touched the file.
2. Compresses the year list into a compact range string
   (e.g. "2023-2024,2026").
3. Rewrites the copyright header line in-place.

Usage:
    python scripts/update_copyrights.py          # dry-run (default)
    python scripts/update_copyrights.py --write   # apply changes

Run from anywhere inside the repository.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Set


# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #

# Directories to scan (relative to repo root)
SCAN_DIRS = ["src/sparkx", "tests"]

# Regex that matches the existing copyright line
COPYRIGHT_RE = re.compile(
    r"^(?P<prefix>#\s+Copyright \(c\) )(?P<years>[0-9, -]+)$"
)


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #


def repo_root() -> Path:
    """Return the top-level directory of the git repository."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def commit_years_for_file(filepath: Path) -> Set[int]:
    """Return the set of years in which *filepath* was modified (per git)."""
    result = subprocess.run(
        ["git", "log", "--follow", "--format=%ad", "--date=format:%Y", "--", str(filepath)],
        capture_output=True,
        text=True,
        check=True,
    )
    years: Set[int] = set()
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line:
            years.add(int(line))
    return years


def compress_years(years: Set[int]) -> str:
    """
    Compress a set of years into a human-readable range string.

    Examples
    --------
    >>> compress_years({2023, 2024, 2025})
    '2023-2025'
    >>> compress_years({2023, 2024, 2026})
    '2023-2024,2026'
    >>> compress_years({2024})
    '2024'
    """
    if not years:
        return ""
    sorted_years = sorted(years)
    ranges: List[str] = []
    start = prev = sorted_years[0]
    for y in sorted_years[1:]:
        if y == prev + 1:
            prev = y
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = y
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def collect_py_files(root: Path) -> List[Path]:
    """Collect all .py files under the configured scan directories."""
    files: List[Path] = []
    for scan_dir in SCAN_DIRS:
        target = root / scan_dir
        if not target.is_dir():
            continue
        for py_file in sorted(target.rglob("*.py")):
            files.append(py_file)
    return files


def update_copyright_in_file(
    filepath: Path, years_str: str, *, write: bool
) -> bool:
    """
    Update the copyright line in *filepath*.

    Returns True if the file was (or would be) changed.
    """
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    changed = False
    for i, line in enumerate(lines):
        m = COPYRIGHT_RE.match(line.rstrip("\n\r"))
        if m:
            old_years = m.group("years").strip()
            if old_years != years_str:
                new_line = f"{m.group('prefix')}{years_str}\n"
                lines[i] = new_line
                changed = True
            break  # only one copyright line per file

    if changed and write:
        filepath.write_text("".join(lines), encoding="utf-8")

    return changed


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update copyright years in SPARKX .py files from git history."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually modify files (default is dry-run).",
    )
    args = parser.parse_args()

    root = repo_root()
    py_files = collect_py_files(root)

    if not py_files:
        print("No .py files found – is the working directory inside the sparkx repo?")
        sys.exit(1)

    updated = 0
    skipped_no_history = 0
    unchanged = 0

    for fpath in py_files:
        rel = fpath.relative_to(root)
        years = commit_years_for_file(fpath)
        if not years:
            skipped_no_history += 1
            continue

        years_str = compress_years(years)
        changed = update_copyright_in_file(fpath, years_str, write=args.write)

        if changed:
            updated += 1
            action = "updated" if args.write else "would update"
            print(f"  {action}: {rel}  -> Copyright (c) {years_str}")
        else:
            unchanged += 1

    print()
    print(f"Files scanned  : {len(py_files)}")
    print(f"Updated        : {updated}")
    print(f"Already correct: {unchanged}")
    if skipped_no_history:
        print(f"No git history : {skipped_no_history}")
    if not args.write and updated:
        print("\nThis was a dry-run. Re-run with --write to apply changes.")


if __name__ == "__main__":
    main()
