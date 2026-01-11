# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from typing import Callable, Optional, Union, Type, List, Generator, Dict

from mir.config.console import dbuq, nfo
from mir.config.constants import DocStringEntry, ClassMapEntry, import_submodules


def retrieve_diffusers_docstrings(
    package_name: str,
    file_names: list[str],
) -> Generator[DocStringEntry]:
    """Yield (pkg, file, EXAMPLE_DOC_STRING) from a folder or a single file.\n
    :param pkg_name: Package under ``diffusers.pipelines``.\n
    :param file_names: A list of related file names.\n
    :param use_folder: True → treat ``source`` as a folder with ``_import_structure``.\n
    :return: DocString Entry class.\n
    """
    import os
    from importlib import import_module

    module_location: str | None = import_module("diffusers.pipelines").__file__
    module_path = os.path.dirname(module_location)

    for file_name in file_names:
        assert isinstance(file_name, str)
        if file_name == "pipeline_stable_diffusion_xl_inpaint":
            continue

        pkg_path = f"diffusers.pipelines.{package_name}.{file_name}"
        dbuq(pkg_path)

        if os.path.exists(os.path.join(module_path, package_name, f"{file_name}.py")):
            pipe_file = import_submodules(file_name, pkg_path) or import_module(pkg_path) or nfo(f"Failed to import {pkg_path}")
            if doc_string := getattr(pipe_file, "EXAMPLE_DOC_STRING", None):
                yield DocStringEntry(package_name=package_name, file_name=file_name, doc_string=doc_string)
            else:
                nfo(f"Doc string attribute missing for {package_name}/{file_name}")
        else:
            nfo(f"Path not found for {package_name}/{file_name}")

    return


def get_repo_from_class_map(class_map: ClassMapEntry) -> str | None:
    """The name of the repository that is associated with a transformers configuration class
    :param class_map: Transformers class information extracted from dependency
    :returns: A string matching the repo path for the class"""

    import re

    doc_attempt = []
    if hasattr(class_map.config, "forward"):
        doc_attempt = [getattr(class_map.config, "forward")]
    doc_attempt.append(class_map.config)
    for pattern in doc_attempt:
        doc_string = pattern.__doc__
        matches = re.findall(r"\[([^\]]+)\]", doc_string)
        if matches:
            try:
                repo_path = next(iter(snip.strip('"').strip() for snip in matches if "/" in snip))
            except StopIteration as error_log:
                nfo(f"ERROR >>{matches} : LOG >> {error_log}")
                continue
            return repo_path
    return None


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
