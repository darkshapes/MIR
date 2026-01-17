# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable
from importlib import import_module

tag = lambda path: path.rsplit(".", 1)  # noqa
run = lambda parts: getattr(import_module(parts[0]), parts[1])


def get_attribute_chain(root_object: Callable, attribute_path: str):
    """Retrieve a nested attribute from *root_object* using a dot‑separated string.\n
    :param root_object : The object from which the attribute chain will be resolved.
    :param attribute_path : Dot‑separated attribute names, e.g. ``"ops.cnn.yolos"``.
    :returns: The final attribute value reached by following the chain.
    :raises: AttributeError If any part of the chain does not exist on the current object."""

    current = root_object
    for part in attribute_path.split("."):
        current = getattr(current, part)
    return current
