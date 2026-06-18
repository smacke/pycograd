# -*- coding: utf-8 -*-
"""Pytrees: nested list/tuple/dict containers with arrays/scalars/Vars as leaves.

Let parameters be one structured object (a dict of weights, say) instead of a pile
of positional arrays, so ``value_and_grad`` returns gradients with the same shape.
list/tuple/dict (and subclasses) are nodes; everything else is a leaf. (No
namedtuple/custom-node registration; a namedtuple round-trips as a tuple.)
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Optional, Union, cast

from pycograd._typing import Array, Operand
from pycograd.params import Param, ParamDict


class _LeafMarker:
    """Placeholder marking a leaf position in a treedef."""


_LEAF = _LeafMarker()

# A leaf is a Param, a Var, a plain number/array, or None (``None`` appears in a
# gradient pytree at a frozen/non-numeric leaf). A pytree is a leaf or a nested
# list/tuple/dict of pytrees; a treedef mirrors that shape with leaves replaced
# by ``_LeafMarker``. These aliases are referenced at runtime (``cast``), so they
# keep ``Union``/``Optional`` rather than PEP 604 ``|`` (invalid at runtime on 3.9).
Leaf = Optional[Union["Param", Operand]]
PyTree = Union[Leaf, list["PyTree"], tuple["PyTree", ...], dict[str, "PyTree"]]
TreeDef = Union[
    _LeafMarker, list["TreeDef"], tuple["TreeDef", ...], dict[str, "TreeDef"]
]


def tree_flatten(tree: PyTree) -> tuple[list[Leaf], TreeDef]:
    """Return ``(leaves, treedef)`` -- the leaves in a fixed order, plus a
    structure description that ``tree_unflatten`` can rebuild from."""
    leaves: list[Leaf] = []

    def build(node: PyTree) -> TreeDef:
        if isinstance(node, list):
            return [build(child) for child in node]
        if isinstance(node, tuple):
            return tuple(build(child) for child in node)
        if isinstance(node, dict):
            built = {key: build(node[key]) for key in sorted(node)}
            # Preserve a dict subtype (e.g. ParamDict) so it survives a round trip.
            return ParamDict(built) if isinstance(node, ParamDict) else built
        leaves.append(node)
        return _LEAF

    return leaves, build(tree)


def tree_unflatten(treedef: TreeDef, leaves: Iterable[Leaf]) -> PyTree:
    """Rebuild a pytree of the given ``treedef`` from a flat ``leaves`` iterable."""
    it = iter(leaves)

    def build(td: TreeDef) -> PyTree:
        if isinstance(td, list):
            return [build(child) for child in td]
        if isinstance(td, tuple):
            return tuple(build(child) for child in td)
        if isinstance(td, dict):
            built = {key: build(td[key]) for key in td}
            return ParamDict(built) if isinstance(td, ParamDict) else built
        return next(it)

    return build(treedef)


def tree_leaves(tree: PyTree) -> list[Leaf]:
    return tree_flatten(tree)[0]


def tree_structure(tree: PyTree) -> TreeDef:
    """The treedef of ``tree`` -- equal (``==``) iff two trees have the same shape."""
    return tree_flatten(tree)[1]


def tree_map(func: Callable[..., Leaf], tree: PyTree, *rest: PyTree) -> PyTree:
    """Apply ``func`` leafwise across one or more same-structured pytrees."""
    leaves, treedef = tree_flatten(tree)
    rest_leaves = [tree_flatten(other)[0] for other in rest]
    return tree_unflatten(treedef, [func(*xs) for xs in zip(leaves, *rest_leaves)])


def _sgd_step(p: Leaf, g: Leaf, lr: float) -> Leaf:
    """One leafwise SGD step, preserving ``Param`` wrappers and skipping anything
    with no gradient (``g is None`` -- a frozen or non-numeric leaf)."""
    if isinstance(p, Param):
        if not p.trainable or g is None:
            return p  # frozen / no gradient: held fixed, wrapper preserved
        return replace(p, value=cast(Array, p.value) - lr * cast(Array, g))
    if g is None:
        return p
    return cast(Array, p) - lr * cast(Array, g)


def sgd_update(params: PyTree, grads: PyTree, lr: float) -> PyTree:
    """One SGD step over an arbitrary param pytree: ``p <- p - lr * g`` leafwise.

    Both pytrees must share a structure (e.g. the gradient pytree that
    ``value_and_grad`` returns for ``params``); the result has the same structure.
    ``Param`` leaves are carried through as ``Param``s -- frozen ones (gradient
    ``None``) are left untouched, trainable ones are stepped in place.
    """
    return tree_map(lambda p, g: _sgd_step(p, g, lr), params, grads)
