from __future__ import annotations

import argparse
import pathlib
import sys

from .checker import Finding, check_source


def _format(path: pathlib.Path, finding: Finding) -> str:
    return f"{path}:{finding.line}:{finding.column + 1}: {finding.code} {finding.message}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check OpenCV image-channel policy")
    parser.add_argument("paths", nargs="*", type=pathlib.Path)
    args = parser.parse_args(argv)

    failed = False
    for path in args.paths:
        try:
            findings = check_source(path.read_text(encoding="utf-8"))
        except SyntaxError as error:
            print(f"{path}:{error.lineno}:{error.offset}: invalid Python syntax: {error.msg}", file=sys.stderr)
            failed = True
            continue
        for finding in findings:
            print(_format(path, finding), file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
