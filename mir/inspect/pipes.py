# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import List, Optional


def show_shared_hyperparameters(parameter_filter: Optional[str] = None) -> List[str]:
    """Show all config classes in the Transformer package with the specified init annotation\n
    :param from_match: Narrow the classes to only those with an exact key inside
    :return: A list of all Classes"""
    from mir.inspect.metadata import map_transformers_classes
    from mir.config.constants import extract_init_parameters

    transformers_data = map_transformers_classes()
    config_data = []
    for entry in transformers_data:
        if parameter_filter:
            segments = extract_init_parameters(module=entry.config, package_name="transformers")
            if parameter_filter in list(segments):
                config_data.append(entry.config)
        else:
            config_data.append(entry.config)
    return config_data


def get_class_parent_folder(class_name: str, pkg_name: str) -> List[str]:
    """Retrieve the folder path within a class. Only returns if it is a valid path in the system (formerly seek_class_path)\n
    ### NOTE: in most cases `__module__` makes this redundant
    :param class_name: The internal name for the model in the third-party API.
    :param pkg_name: The API Package
    :return: A list corresponding to the path of the model, or None if not found
    :raises KeyError: for invalid pkg_name
    """
    from mir.config.console import dbuq
    from mir.inspect.classes import resolve_code_names, extract_init_params

    pkg_name = pkg_name.lower()
    if pkg_name == "diffusers":
        parent_folder: List[str] = resolve_code_names(class_name=class_name, pkg_name=pkg_name, path_format=True)
        if not parent_folder or not parent_folder[-1].strip():
            dbuq("Data not found for", " class_name = {class_name},pkg_name = {pkg_name},{parent_folder} = parent_folder")
            return None
    elif pkg_name == "transformers":
        module_path = extract_init_params(class_name, "transformers").get("config")
        parent_folder = module_path[:3]
    return parent_folder
