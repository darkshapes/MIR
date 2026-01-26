# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# 模块发现和解构

import inspect
from importlib import import_module
from inspect import getmro
from types import ModuleType
from typing import Callable

tag = lambda path: path.rsplit(".", 1)  # noqa
run = lambda parts: getattr(import_module(parts[0]), parts[1])


def get_attribute_chain(root_object: Callable | ModuleType, attribute_path: str) -> Callable | ModuleType:
    """Retrieve a nested attribute from *root_object* using a dot-separated string.\n
    :param root_object : The object from which the attribute chain will be resolved.
    :param attribute_path : Dot-separated attribute names, e.g. ``"ops.cnn.yolos"``.
    :returns: The final attribute value reached by following the chain.
    :raises: AttributeError If any part of the chain does not exist on the current object."""
    current = root_object
    for part in attribute_path.split("."):
        current = getattr(current, part)
    return current


def get_import_chain(class_path: str) -> Callable | ModuleType:
    """Retrieve a class object from dot-separated string reference.\n
    :param class_path : The object from which the attribute chain will be resolved.
    :returns: The final imported object reached by following the chain.
    :raises: AttributeError If any part of the chain does not exist on the current object."""
    library_name = class_path.split(".")[0]
    attribute_path = class_path.replace(library_name + ".", "")
    library = import_module(library_name)
    path_chain = get_attribute_chain(library, attribute_path)
    return path_chain


def migrations(repo_path: str) -> str:
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


def extract_subclass_data(package_name: str, base_class_name: str, all: bool = False) -> dict[str, Callable] | None:
    """Extracts subclasses from a package that inherit from a specified base class.\n
    :param package_name: Name of the package to search
    :param base_class_name: Name of the base class to inherit from
    :return: Dictionary mapping fully qualified class names to class objects"""

    from importlib import import_module
    from pkgutil import walk_packages

    subclasses = {}
    root_pkg = import_module(package_name)
    if package_path := getattr(root_pkg, "__path__", root_pkg.__all__):
        for finder, module_name, is_pkg in walk_packages(package_path, root_pkg.__name__ + "."):
            try:
                module = import_module(module_name)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue

            for name, obj in module.__dict__.items():
                print(obj)
                if not isinstance(obj, type):
                    obj = import_module(obj, root_pkg.__module__)
                if obj.__module__ != module_name:
                    continue
                try:
                    bases = getmro(obj)[1:]  # skip the class itself
                except ValueError:
                    continue
                for base in bases:
                    if base.__name__ == base_class_name:
                        fqcn = f"{module_name}.{name}"
                        subclasses[fqcn] = obj
                        break

    return subclasses


def get_source_of(class_obj: Callable) -> list[str]:
    """Retrieve the source lines of a class definition.\n
    :params class_obj: The class object whose source is to be read.
    :return: A list of source lines from the class's file."""
    from mir.lookups import get_import_chain

    module = class_obj.__module__
    chain = get_import_chain(module)
    file_path_named: str = chain.__file__  # type: ignore
    with open(file_path_named) as opened_file:
        file_lines = opened_file.readlines()
    return file_lines


def nn_source_tree(file_lines: list[str]) -> dict[str, str] | None:
    """Parse a list of source lines to locate a ModuleList call.\n
    :params file_lines: Lines of source code to analyze.
    :return: Mapping of class name to the call string if found, otherwise None."""

    import ast

    target = "ModuleList("
    tree = ast.parse("".join(file_lines))
    node_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for current_node in ast.walk(node):
                if isinstance(current_node, ast.Call) and isinstance(current_node.func, ast.Attribute):
                    if current_node.func.attr == "ModuleList":
                        line_number = current_node.lineno
                        source_code = file_lines[line_number - 1].strip()
                        if source_code.endswith(target):
                            source_code = file_lines[line_number].strip()
                        if class_name := list(name for name in node_names if name + "(" in source_code):
                            layer_data = source_code.rsplit("range", 1)[-1]
                            layer_data = layer_data.split(")", 1)[0].split(".", 1)[1]
                            return {"class_name": class_name[0], "config_attribute": layer_data}


def find_nn_modules(module: Callable, prefix: str = ""):
    """
    Traverse through the module and its children, collecting all nn.Module instances.

    Args:
        module (torch.nn.Module): The module to inspect.
        prefix (str): The prefix for the module names during recursion.

    Returns:
        List[torch.nn.Module]: A list of all nn.Module tuple instances found.
    """
    from torch import nn

    nn_modules = {}
    library = module.__module__.split(".", 1)[0]
    module_path = get_import_chain(module.__module__)
    for attribute in sorted(dir(module_path)):
        if attribute.startswith("_"):
            continue
        attribute_object = getattr(module_path, attribute)
        if isinstance(attribute_object, type) and library in attribute_object.__module__ and nn.Module in getmro(attribute_object):
            nn_modules.setdefault((attribute, attribute_object))
    return nn_modules


def find_config_classes(parameter_filter: str) -> list[str]:
    """Show all config classes in the Transformer package with the specified init annotation\n
    :param from_match: Narrow the classes to only those with an exact key inside
    :return: A list of all Classes"""

    from mir.gatherers.transformers import CONFIG_MAPPING

    # filler = ["bool", "int", "float", "complex", "str", "list", "tuple", "dict", "set"]
    config_data = []
    for config_class in CONFIG_MAPPING.values():
        if isinstance(config_class, tuple):
            config_class = config_class[0]
        signature = inspect.signature(config_class.__init__)
        if parameter_filter in list(signature.parameters):
            config_data.append(config_class.__name__)
    return config_data
