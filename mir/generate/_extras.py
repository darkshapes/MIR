# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable, Dict, List, Optional, Union

from mir import NFO
from mir.generate.from_module import import_object_named, show_path_for
from mir.generate.tasks import TaskAnalyzer


def _class_parent(code_name: str, pkg_name: str) -> Optional[List[str]]:
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


def _extract_inherited_classes(model_class: Union[Callable, str], pkg_name: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
    """Strips <class> tags from module's base classes and extracts inherited class members.\n
    If `module` is a string, it requires the `library` argument to convert it into a callable.\n
    :param module: A module or string representing a module.
    :param library: Library name required if `module` is a string. Defaults to None.
    :returns: Mapping indices to class path segments, or None if invalid input."""

    if isinstance(model_class, str):
        if not pkg_name:
            NFO("Provide a library type argument to process strings")
            return None
        model_class = import_object_named(model_class, pkg_name)
    signature = model_class.__bases__
    class_names = []
    for index, class_annotation in enumerate(signature):
        tag_stripped = str(class_annotation)[8:-2]
        module_segments = tag_stripped.split(".")
        class_names.append(module_segments)
    return class_names


def _trace_classes(pipe_class: str, pkg_name: str) -> Dict[str, List[str]]:
    """Retrieve all compatible pipe forms\n
    NOTE: Mainly for Diffusers
    :param pipe_class: Origin pipe
    :param pkg_name: Dependency package
    :return: A dictionary of pipelines"""

    related_pipes = []
    code_name = show_path_for(pipe_class, pkg_name)
    if pkg_name == "diffusers":
        related_pipe_class_name = pipe_class
    else:
        related_pipe_class_name = None
    related_pipes: list[str] = TaskAnalyzer.show_diffusers_tasks(code_name=code_name, class_name=related_pipe_class_name)
    # for i in range(len(auto_tasks)):
    #     auto_tasks.setdefault(i, revealed_tasks[i])
    parent_folder = class_parent(code_name, pkg_name)
    if pkg_name == "diffusers":
        pkg_folder = import_object_named(parent_folder[0], ".".join(parent_folder))
    else:
        pkg_folder = import_object_named("__init__", ".".join(parent_folder[:-1]))
    if hasattr(pkg_folder, "_import_structure"):
        related_pipes.extend(next(iter(x)) for x in pkg_folder._import_structure.values())
    related_pipes = set(related_pipes)
    related_pipes.update(tuple(x) for x in _extract_inherited_classes(model_class=pipe_class, pkg_name=pkg_name))
    return related_pipes


def _show_shared_hyperparameters(parameter_filter: Optional[str] = None) -> List[str]:
    """Show all config classes in the Transformer package with the specified init annotation\n
    :param from_match: Narrow the classes to only those with an exact key inside
    :return: A list of all Classes"""
    from mir.config.constants import extract_init_parameters
    from mir.inspect.metadata import find_transformers_classes

    transformers_data = find_transformers_classes()
    config_data = []
    for entry in transformers_data:
        if parameter_filter:
            segments = extract_init_parameters(module=entry.config, package_name="transformers")
            if parameter_filter in list(segments):
                config_data.append(entry.config)
        else:
            config_data.append(entry.config)
    return config_data


def _get_class_parent_folder(class_name: str, pkg_name: str) -> List[str]:
    """Retrieve the folder path within a class. Only returns if it is a valid path in the system (formerly seek_class_path)\n
    ### NOTE: in most cases `__module__` makes this redundant
    :param class_name: The internal name for the model in the third-party API.
    :param pkg_name: The API Package
    :return: A list corresponding to the path of the model, or None if not found
    :raises KeyError: for invalid pkg_name
    """
    from mir.config.console import dbuq
    from mir.config.constants import extract_init_parameters
    from mir.inspect.classes import resolve_code_names

    pkg_name = pkg_name.lower()
    if pkg_name == "diffusers":
        parent_folder: List[str] = resolve_code_names(class_name=class_name, pkg_name=pkg_name, path_format=True)
        if not parent_folder or not parent_folder[-1].strip():
            dbuq("Data not found for", " class_name = {class_name},pkg_name = {pkg_name},{parent_folder} = parent_folder")
            return None
    elif pkg_name == "transformers":
        print(class_name)
        module_path = extract_init_parameters(class_name, "transformers")
        print(module_path)
        config = str(module_path.get("config"))
        print(config)
        config = config.split(": ")[-1].split(".")
        parent_folder = config[:3]
    return parent_folder


def _class_to_mir_tag(mir_db: Dict[str, str], code_name: str) -> Optional[str]:
    """Converts a class identifier to its corresponding MIR tag.\n
    :param mir_db: A dictionary mapping series-compatibility pairs to their respective data.
    :param code_name: The Transformers class identifier to convert.
    :return: An optional list containing the series and compatibility if found, otherwise None."""

    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

    from mir.config.constants import TEMPLATE

    template_data = TEMPLATE["arch"]["transformer"]

    for series, compatibility_data in mir_db.database.items():
        if any([template for template in template_data if template in series.split(".")[1]]):
            for compatibility, field_data in compatibility_data.items():
                if code_name == series.split(".")[2]:
                    return [series, compatibility]

                class_name = MODEL_MAPPING_NAMES.get(code_name, False)
                if not class_name:  # second pass without separators
                    recoded_mapping = {code.replace("-", "").replace("_", ""): model for code, model in MODEL_MAPPING_NAMES.items()}
                    class_name = recoded_mapping.get(code_name, False)
                    if not class_name:
                        return None
                pkg_data = field_data.get("pkg")
                if pkg_data:
                    for _, pkg_type_data in pkg_data.items():
                        maybe_class = pkg_type_data.get("transformers")
                        if maybe_class == class_name:
                            return [series, compatibility]
    return None


def tag_transformers_model(repo_path: str, class_name: str, addendum: dict | None = None) -> tuple[str, str, str | dict[str, dict]]:
    """Convert model repo paths to MIR tags, classifying by feature\n
    :param name: Repo path
    :param class_name: The HF transformers class for the model
    :return: A segmented MIR tag useful for appending index entries"""

    from mir.config.constants import extract_init_parameters

    annotations = extract_init_parameters(class_name.replace("Model", "Config"), "transformers")
    if not annotations:
        class_name = class_name.replace("Config", "Model")
        annotations = extract_init_parameters(class_name, "transformers")
    if not annotations:
        raise TypeError("No mode type returned")
    if "Bert" in class_name:
        print(annotations)
    mir_prefix = mir_prefix_from_forward_pass(True, **annotations)
    base_series, base_comp = tag_model_from_repo(repo_path)
    if not addendum:
        return mir_prefix, base_series, base_comp
    else:
        mir_prefix = f"info.{mir_prefix}"
    return mir_prefix, base_series, {base_comp: addendum}
