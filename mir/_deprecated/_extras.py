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


# def extract_model_data(self,pipe_name, file_name: str) -> dict | None:
#     migrated_pipes = MIGRATIONS["migrated_pipes"]
#     pkg_path = f"diffusers.pipelines.{pipe_name}.{file_name}"
#     pipe_file: Callable = import_object_named(file_name, pkg_path) or import_module(pkg_path)
#     if pipe_file and (doc_string := getattr(pipe_file, "EXAMPLE_DOC_STRING", None)): #where pipe class and repo are
#         docstrings= DocStringEntry(package_name=pipe_name, file_name=file_name, pipe_module=pipe_file, doc_string=doc_string)
#         DocStringParser(doc_string=docstrings.doc_string)
#         self.parsed_docs.pipe_repo = migrated_pipes.get(self.parsed_docs.pipe_class, self.parsed_docs.pipe_repo)
#         model = import_object_named(parsed_data.pipe_class, docstrings.pipe_module.__name__)
#         model_data = show_init_fields_for(model,"diffusers")
#         return {"model_params": model_data}


#   for pipe_name in IMPORT_STRUCTURE.keys():
#             if pipe_name not in exclusion_list and (import_name := getattr(diffusers_pipelines, str(pipe_name))):
#                 file_specific = uncommon_naming.get(pipe_name, pipe_name)
#                 file_names:list[str] = [getattr(import_name, "_import_structure", {})] or [f"pipeline_{file_specific}"]
#                 for file_name in file_names:
#                     if not file_name in exclusion_list or not (model_data := self.extract_model_data(pipe_name, file_name)):
#                         continue
#                     if not (prepared_data := PrepareData( **model_data)):
#                         continue
# else:
# continue


# def show_path_for(code_name: str, pkg_name: str) -> list[str] | str | None:
#     """Retrieve the folder path within a class. Only returns if it is a valid path in the system\n
#     ### NOTE: in most cases `__module__` makes this redundant
#     :param code_name: The internal name for the model in the third-party API.
#     :param pkg_name: The API Package
#     :return: A list corresponding to the path of the model, or None if not found
#     :raises KeyError: for invalid pkg_name
#     """

#     pkg_paths = {
#         "diffusers": "pipelines",
#         "transformers": "models",
#     }
#     folder_name = code_name.replace("-", "_")
#     pkg_name = pkg_name.lower()
#     folder_path = pkg_paths[pkg_name]
#     package_obj = import_module(pkg_name)
#     folder_path_named = [folder_path, folder_name]
#     pkg_folder = os.path.dirname(getattr(package_obj, "__file__"))
#     # dbuq(os.path.exists(os.path.join(pkg_folder, *folder_path_named)))
#     if os.path.exists(os.path.join(pkg_folder, *folder_path_named)) is True:
#         import_path = [pkg_name]
#         import_path.extend(folder_path_named)
#         return import_path


# def get_internal_name_for(module_name: str | Type | None = None, pkg_name: str = "transformers", path_format: bool | None = False) -> list[str] | str | None:
#     """Reveal code names for class names from Diffusers or Transformers (formerly get code names)\n
#     :param class_name: To return only one class, defaults to None
#     :param pkg_name: optional field for library, defaults to "transformers"
#     :param path_format: Retrieve just the code name, or the full module path and code name within the package
#     :return: A list of all code names, or the one corresponding to the provided class"""
#     from mir.generate.diffusers import IMPORT_STRUCTURE
#     from mir.generate.transformers import MODEL_MAPPING_NAMES

#     package_imports = IMPORT_STRUCTURE if pkg_name == "diffusers" else MODEL_MAPPING_NAMES
#     pkg_name = pkg_name.lower()
#     MAPPING_NAMES: dict[str, str] = import_object_named(*package_imports[pkg_name])
#     if module_name:
#         if isinstance(module_name, Type):
#             module_name = module_name.__name__
#         code_name = next(iter(key for key, value in MAPPING_NAMES.items() if module_name in str(value)), "")
#         return show_path_for(code_name, pkg_name) if path_format else code_name.replace("_", "-")
#     return list(MAPPING_NAMES)
