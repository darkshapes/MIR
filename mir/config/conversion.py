# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable, Optional, Union, Type, List, Iterator, Tuple, Dict
from mir.config.logs import dbuq, nfo


def import_submodules(module_name: str, pkg_name_or_abs_path: str) -> Optional[Callable]:
    """Convert two strings into a callable function or property\n
    :param module: The name of the module to import
    :param library_path: Base package for the module
    :return: The callable attribute or property
    """
    from importlib import import_module

    module = module_name.strip()
    library = pkg_name_or_abs_path.strip()
    base_library = import_module(library, module)
    try:
        module = getattr(base_library, module)
        return module
    except AttributeError:  # as error_log:
        # dbuq(error_log)
        return base_library


def code_name_to_class_name(
    code_name: Optional[Union[str, Type]] = None,
    pkg_name: Optional[str] = "transformers",
) -> Union[List[str], str]:
    """Fetch class names from code names from Diffusers or Transformers\n
    :param class_name: To return only one class, defaults to None
    :param pkg_name: optional field for library, defaults to "transformers"
    :return: A list of all code names, or the one corresponding to the provided class"""
    from mir.config.constants import package_map

    pkg_name = pkg_name.lower()
    MAPPING_NAMES = import_submodules(*package_map[pkg_name])
    if code_name:
        return MAPPING_NAMES.get(code_name)
    return list(MAPPING_NAMES.keys())


def pkg_path_to_docstring(pkg_name: str, folder_path: bool) -> Iterator[Tuple[str, str, str]]:
    """Processes package folder paths to yield example doc strings if available.\n
    :param pkg_name: The name of the package under diffusers.pipelines.
    :param file_specific: A flag indicating whether processing is specific to certain files.
    :yield: A tuple containing (pkg_name, file_name, EXAMPLE_DOC_STRING) if found.
    """
    import os
    from importlib import import_module

    file_names = list(getattr(folder_path, "_import_structure").keys())
    module_path = os.path.dirname(import_module("diffusers.pipelines").__file__)
    for file_name in file_names:
        if file_name == "pipeline_stable_diffusion_xl_inpaint":
            continue
        try:
            pkg_path = f"diffusers.pipelines.{str(pkg_name)}.{file_name}"
            dbuq(pkg_path)
            path_exists = os.path.exists(os.path.join(module_path, pkg_name, file_name + ".py"))
            if path_exists:
                print(f"file_name, pkg_path): {file_name, pkg_path}")
                pipe_file = import_submodules(file_name, pkg_path)
        except ModuleNotFoundError:
            if pkg_name != "skyreels_v2":
                nfo(f"Module Not Found for {pkg_name}")
            pipe_file = None

        try:
            if pipe_file and hasattr(pipe_file, "EXAMPLE_DOC_STRING"):
                yield (pkg_name, file_name, pipe_file.EXAMPLE_DOC_STRING)
            else:
                if path_exists:
                    pipe_file = import_module(pkg_path)
        except (ModuleNotFoundError, AttributeError):
            if pkg_name != "skyreels_v2":
                nfo(f"Doc String Not Found for {pipe_file} {pkg_name}")


def file_name_to_docstring(pkg_name: str, file_specific: bool) -> Iterator[Tuple[str, str, str]]:
    """Processes package using file name to yield example doc strings if available.\n
    :param pkg_name: The name of the package under diffusers.pipelines.
    :param file_specific: A flag indicating whether processing is specific to certain files.
    :yield: A tuple containing (pkg_name, file_name, EXAMPLE_DOC_STRING) if found.
    """
    from importlib import import_module

    file_name = f"pipeline_{file_specific}"
    try:
        pkg_path = f"diffusers.pipelines.{str(pkg_name)}"
        pipe_file = import_submodules(file_name, pkg_path)
    except ModuleNotFoundError:
        if pkg_name != "skyreels_v2":
            nfo(f"Module Not Found for {pkg_name}")
        pipe_file = None
    try:
        if pipe_file and hasattr(pipe_file, "EXAMPLE_DOC_STRING"):
            yield (pkg_name, file_name, pipe_file.EXAMPLE_DOC_STRING)
        else:
            pipe_file = import_module(pkg_path)

    except AttributeError:
        if pkg_name != "skyreels_v2":
            nfo(f"Doc String Not Found for {pipe_file} {pkg_name}")


def class_to_mir_tag(mir_db: Dict[str, str], code_name: str) -> Optional[str]:
    """Converts a class identifier to its corresponding MIR tag.\n
    :param mir_db: A dictionary mapping series-compatibility pairs to their respective data.
    :param code_name: The Transformers class identifier to convert.
    :return: An optional list containing the series and compatibility if found, otherwise None."""
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
    from mir.config.constants import template

    template_data = template["arch"]["transformer"]

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


def slice_number(text: str) -> Union[int, float, str]:
    """Separate a numeral value appended to a string\n
    :return: Converted value as int or float, or unmodified string
    """
    for index, char in enumerate(text):  # Traverse forwards
        if char.isdigit():
            numbers = text[index:]
            if "." in numbers:
                return float(numbers)
            try:
                return int(numbers)
            except ValueError:
                return numbers
    return text
