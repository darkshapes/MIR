# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# 模块发现和解构

import inspect
import os
from importlib import import_module
from typing import Callable, Type

from mir import NFO
from mir.generate import REGEX
from mir.generate.diffusers import IMPORT_STRUCTURE
from mir.generate.transformers import MODEL_MAPPING_NAMES


def import_object_named(module: str, pkg_name_or_abs_path: str) -> Callable | None:
    """Convert two strings into a callable function or property\n
    :param module: The name of the module to import
    :param library_path: Base package for the module
    :return: The callable attribute or property
    """

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


def show_path_for(code_name: str, pkg_name: str) -> list[str] | str | None:
    """Retrieve the folder path within a class. Only returns if it is a valid path in the system\n
    ### NOTE: in most cases `__module__` makes this redundant
    :param code_name: The internal name for the model in the third-party API.
    :param pkg_name: The API Package
    :return: A list corresponding to the path of the model, or None if not found
    :raises KeyError: for invalid pkg_name
    """

    pkg_paths = {
        "diffusers": "pipelines",
        "transformers": "models",
    }
    folder_name = code_name.replace("-", "_")
    pkg_name = pkg_name.lower()
    folder_path = pkg_paths[pkg_name]
    package_obj = import_module(pkg_name)
    folder_path_named = [folder_path, folder_name]
    pkg_folder = os.path.dirname(getattr(package_obj, "__file__"))
    # dbuq(os.path.exists(os.path.join(pkg_folder, *folder_path_named)))
    if os.path.exists(os.path.join(pkg_folder, *folder_path_named)) is True:
        import_path = [pkg_name]
        import_path.extend(folder_path_named)
        return import_path


def get_internal_name_for(module_name: str | Type | None = None, pkg_name: str = "transformers", path_format: bool | None = False) -> list[str] | str | None:
    """Reveal code names for class names from Diffusers or Transformers (formerly get code names)\n
    :param class_name: To return only one class, defaults to None
    :param pkg_name: optional field for library, defaults to "transformers"
    :param path_format: Retrieve just the code name, or the full module path and code name within the package
    :return: A list of all code names, or the one corresponding to the provided class"""

    package_imports = IMPORT_STRUCTURE if pkg_name == "diffusers" else MODEL_MAPPING_NAMES
    pkg_name = pkg_name.lower()
    MAPPING_NAMES: dict[str, str] = import_object_named(*package_imports[pkg_name])
    if module_name:
        if isinstance(module_name, Type):
            module_name = module_name.__name__
        code_name = next(iter(key for key, value in MAPPING_NAMES.items() if module_name in str(value)), "")
        return show_path_for(code_name, pkg_name) if path_format else code_name.replace("_", "-")
    return list(MAPPING_NAMES)


def to_domain_tag(transformers: bool = False, **kwargs):
    """Set type of MIR prefix depending on model type\n
    :param transformers: Use transformers data instead of diffusers data, defaults to False
    :raises ValueError: Model type not detected
    :return: MIR prefix based on model configuration"""

    data = REGEX

    if transformers:
        flags = data["arch"]["transformer"]  # pylint:disable=unsubscriptable-object
    else:
        flags = data["arch"]["diffuser"]  # pylint:disable=unsubscriptable-object
    for mir_prefix, key_match in flags.items():
        if any(kwargs.get(param, None) for param in key_match):
            return mir_prefix
    return None
