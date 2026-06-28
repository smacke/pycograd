# -*- coding: utf-8 -*-
"""Parse the package version (computed live from git tags by versioneer) into
a ``(major, minor, patch)`` tuple -- used by ``scripts/bump-version.py``.
"""
from __future__ import annotations

import re

from pycograd import __version__


def make_version_tuple(version: str | None = None) -> tuple[int, int, int]:
    version = version if version is not None else __version__
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return (0, 0, 0)
    major, minor, patch = (int(part) for part in match.groups())
    return (major, minor, patch)
