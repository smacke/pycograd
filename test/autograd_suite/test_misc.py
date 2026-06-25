# -*- coding: utf-8 -*-
"""Ported placeholder for autograd ``tests/test_misc.py`` (MIT).

Skipped wholesale: autograd-internal. It tests ``autograd.misc.const_graph`` (graph caching)
and ``autograd.misc.flatten`` (a ravel-to-vector utility). pycograd has neither: it has no
const-graph caching primitive, and its ``tree_flatten`` returns ``(leaves, treedef)`` rather
than a flat vector + unflattener. See REPORT.md (semantic divergences).
"""
import pytest

pytest.skip(
    "autograd-internal: const_graph / flatten utilities have no pycograd analog",
    allow_module_level=True,
)
