# ### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
# ### <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""

# pylint:disable=protected-access

from typing import Callable, Dict, List, Optional, Union, Type
from mir.config.conversion import import_submodules
from mir.config.logging import nfo


def resolve_import_path(code_name: str, pkg_name: str) -> Optional[List[str]]:
    """Retrieve the folder path within a class. Only returns if it is a valid path in the system\n
    ### NOTE: in most cases `__module__` makes this redundant
    :param code_name: The internal name for the model in the third-party API.
    :param pkg_name: The API Package
    :return: A list corresponding to the path of the model, or None if not found
    :raises KeyError: for invalid pkg_name
    """
    import os
    from importlib import import_module

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


def resolve_class_names(class_name: Optional[Union[str, Type]] = None, pkg_name: Optional[str] = "transformers", path_format: Optional[bool] = False) -> Union[List[str], str]:
    """Reveal code names for class names from Diffusers or Transformers\n
    :param class_name: To return only one class, defaults to None
    :param pkg_name: optional field for library, defaults to "transformers"
    :param path_format: Retrieve just the code name, or the full module path and code name within the package
    :return: A list of all code names, or the one corresponding to the provided class"""

    package_map = {
        "diffusers": ("_import_structure", "diffusers.pipelines"),
        "transformers": ("MODEL_MAPPING_NAMES", "transformers.models.auto.modeling_auto"),
    }
    pkg_name = pkg_name.lower()
    MAPPING_NAMES = import_submodules(*package_map[pkg_name])
    if class_name:
        if isinstance(class_name, Type):
            class_name = class_name.__name__
        code_name = next(iter(key for key, value in MAPPING_NAMES.items() if class_name in str(value)), "")
        return resolve_import_path(code_name, pkg_name) if path_format else code_name.replace("_", "-")
    return list(MAPPING_NAMES)


def extract_inherited_classes(model_class: Union[Callable, str], pkg_name: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
    """Strips <class> tags from module's base classes and extracts inherited class members.\n
    If `module` is a string, it requires the `library` argument to convert it into a callable.\n
    :param module: A module or string representing a module.
    :param library: Library name required if `module` is a string. Defaults to None.
    :returns: Mapping indices to class path segments, or None if invalid input."""

    if isinstance(model_class, str):
        if not pkg_name:
            nfo("Provide a library type argument to process strings")
            return None
        model_class = import_submodules(model_class, pkg_name)
    signature = model_class.__bases__
    class_names = []
    for index, class_annotation in enumerate(signature):
        tag_stripped = str(class_annotation)[8:-2]
        module_segments = tag_stripped.split(".")
        class_names.append(module_segments)
    return class_names


def extract_init_params(module: Union[Callable, str], pkg_name: Optional[str] = None) -> Dict[str, List[str]]:
    """Pick apart a Diffusers or Transformers pipeline class and find its constituent parts\n
    :param module: Origin pipeline as a class or as a string
    :param library: name of a library to import the class from, only if a string is provided
    :return: Dictionary of sub-classes from the `module`"""

    import inspect

    if pkg_name and isinstance(module, str):
        module = import_submodules(module, pkg_name)
    signature = inspect.signature(module.__init__)
    class_names = {}
    for folder, param in signature.parameters.items():
        if folder != "self":
            sub_module = str(param.annotation).split("'")
            if len(sub_module) > 1 and sub_module[1] not in [
                "bool",
                "int",
                "float",
                "complex",
                "str",
                "list",
                "tuple",
                "dict",
                "set",
            ]:
                class_names.setdefault(folder, sub_module[1].split("."))
    return class_names


# def pull_weight_map(repo_id: str, arch: str) -> Dict[str, str]:
#     from nnll.download.hub_cache import download_hub_file

#     model_file = download_hub_file(
#         repo_id=f"{repo_id}/tree/main/{arch}",
#         source="huggingface",
#         file_name="diffusion_pytorch_model.safetensors.index.json",
#         local_dir=".tmp",
#     )
