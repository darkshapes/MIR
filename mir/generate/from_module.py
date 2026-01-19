# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# 模块发现和解构

import inspect

from importlib import import_module
from typing import Callable


def migrations(repo_path: str):
    """Replaces old organization names in repository paths with new ones.\n
    :param repo_path: Original repository path containing old organization names
    :return: Updated repository path with new organization names"""
    from mir.data import MIGRATIONS

    repo_migrations = MIGRATIONS
    for old_name, new_name in repo_migrations.items():
        if old_name in repo_path:
            repo_path = repo_path.replace(old_name, new_name)
    return repo_path


def import_object_named(module: str, pkg_name_or_abs_path: str) -> Callable | None:
    """Convert two strings into a callable function or property\n
    :param module: The name of the module to import
    :param library_path: Base package for the module
    :return: The callable attribute or property
    """
    from mir import NFO

    module_normalized: str = module.strip()
    library = pkg_name_or_abs_path.strip()
    try:
        base_library = import_module(library, module_normalized)
    except SyntaxError:
        base_library = None
        NFO(f"Syntax error attempting to import {module_normalized}")
    else:
        module_obj = getattr(base_library, module_normalized)
        return module_obj
    return None


def show_init_fields_for(module: Callable | str, package_name: str | None = None, erase: bool = False) -> dict[str, list[str]]:
    """Pick apart a Diffusers or Transformers pipeline class and find its constituent parts\n
    :param module: Origin pipeline as a class or as a string
    :param library: name of a library to import the class from, only if a string is provided
    :return: Dictionary of sub-classes from the `module`"""

    if package_name and isinstance(module, str):
        module_obj: Callable | None = import_object_named(module, package_name)
    else:
        assert isinstance(module, Callable), f"Expected Callable module object, got {module} type {type(module)}"
        module_obj = module
    assert isinstance(module_obj, Callable), f"Expected Callable module object, got {module} type {type(module)}"
    signature = inspect.signature(module_obj.__init__)
    editable_signature = signature.parameters.copy()
    editable_signature.pop("self", None)
    editable_signature.pop("kwargs", None)
    editable_signature.pop("use_cache", None)
    class_names = {}
    if erase:
        for folder, param in editable_signature.items():
            class_names.setdefault(folder, True)
    else:
        for folder, param in editable_signature.items():
            class_names.setdefault(folder, str(param))
    class_names = dict(class_names)

    return class_names
