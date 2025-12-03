# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import List, Optional


def get_transformer_config_classes(parameter_filter: Optional[str] = None) -> List[str]:
    """Show all config classes in the Transformer package with the specified init annotation\n
    :param from_match: Narrow the classes to only those with an exact key inside
    :return: A list of all Classes"""
    from mir.inspect.metadata import gather_transformers_metadata
    from mir.inspect.classes import extract_init_params

    transformers_data = gather_transformers_metadata()
    config_data = []
    for model_path in list(transformers_data.values()):
        config_class = model_path["config"][-1]
        if parameter_filter:
            segments = extract_init_params(config_class, pkg_name="transformers")
            if parameter_filter in list(segments):
                config_data.append(config_class)
        else:
            config_data.append(config_class)
    return config_data


def get_class_parent_folder(class_name: str, pkg_name: str) -> List[str]:
    from mir import dbuq
    from mir.inspect.classes import resolve_class_name, extract_init_params

    pkg_name = pkg_name.lower()
    if pkg_name == "diffusers":
        parent_folder: List[str] = resolve_class_name(class_name=class_name, pkg_name=pkg_name, path_format=True)
        if not parent_folder or not parent_folder[-1].strip():
            dbuq("Data not found for", " class_name = {class_name},pkg_name = {pkg_name},{parent_folder} = parent_folder")
            return None
    elif pkg_name == "transformers":
        module_path = extract_init_params(class_name, "transformers").get("config")
        parent_folder = module_path[:3]
    return parent_folder
