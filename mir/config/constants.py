# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import os
from dataclasses import dataclass, field
from typing import Callable, List

import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_MAPPING_NAMES

from mir.config.json_io import read_json_file
from mir.config.console import nfo


def mapped_cls(model_identifier: str):
    """Get model class from identifier without calling huggingface_hub.\n
    :param model_identifier: Model identifier like "bert-base-uncased" or "gpt2"
    :return: Model class (e.g., BertModel, GPT2Model)
    """
    code_name = model_identifier.split("/")[-1].split("-")[0].lower()

    model_class_name = MODEL_MAPPING_NAMES.get(code_name, None)

    config_class_name = CONFIG_MAPPING_NAMES.get(code_name)
    if config_class_name:
        config_class = getattr(transformers, config_class_name, None)
        if config_class:
            model_class = MODEL_MAPPING.get(config_class, None)
            if model_class:
                if isinstance(model_class, tuple):
                    model_class = model_class[0]
                    return model_class

    normalized = code_name.replace("_", "-")
    if normalized != code_name:
        if model_class_name := MODEL_MAPPING_NAMES.get(normalized, None):
            if isinstance(model_class_name, tuple):
                model_class_name = model_class_name[0]
            return getattr(transformers, model_class_name, None)

    return None


def import_submodules(module_name: str, pkg_name_or_abs_path: str) -> Callable | None:
    """Convert two strings into a callable function or property\n
    :param module: The name of the module to import
    :param library_path: Base package for the module
    :return: The callable attribute or property
    """
    from importlib import import_module

    module = module_name.strip()
    library = pkg_name_or_abs_path.strip()
    try:
        base_library = import_module(library, module)
    except SyntaxError:
        base_library = None
        nfo(f"Syntax error attempting to import {module_name}")
    if module := getattr(base_library, module, None):
        return module
    else:
        nfo("failed to find module {module}")


def extract_init_parameters(module: Callable | str, package_name: str | None = None) -> dict[str, list[str]]:
    """Pick apart a Diffusers or Transformers pipeline class and find its constituent parts (formerly root_class)\n
    :param module: Origin pipeline as a class or as a string
    :param library: name of a library to import the class from, only if a string is provided
    :return: Dictionary of sub-classes from the `module`"""

    import inspect

    if package_name and isinstance(module, str):
        module_obj: Callable = import_submodules(module, package_name)
    else:
        assert isinstance(module, Callable)
        module_obj = module
    signature = inspect.signature(module_obj.__init__)
    class_names = {}
    editable_signature = signature.parameters.copy()
    editable_signature.pop("self", None)
    editable_signature.pop("kwargs", None)
    editable_signature.pop("use_cache", None)
    for folder, param in editable_signature.items():
        class_names.setdefault(folder, True)
    return class_names


@dataclass
class ClassMapEntry:
    """Represents a structured entry of the name of the class and its associated attributes."""

    name: str
    model_name: str
    model: Callable
    config: Callable
    config_params: dict[str, list[str]] = field(init=False, default_factory=lambda: {})
    model_params: dict[str, list[str]] | None = None

    def __post_init__(self):
        if self.model:
            self.model_params = extract_init_parameters(self.model)
        if self.config:
            self.config_params = extract_init_parameters(self.config)


@dataclass
class DocStringEntry:
    """Represents a structured entry of package name, file name, and docstring."""

    package_name: str
    file_name: str
    doc_string: str


class DocParseData:
    pipe_class: str
    pipe_repo: str
    staged_class: str | None = None
    staged_repo: str | None = None

    def __init__(self, pipe_class: str, pipe_repo: str, staged_class: str | None = None, staged_repo: str | None = None):
        self.pipe_class = pipe_class
        self.pipe_repo = pipe_repo
        self.staged_class = staged_class
        self.staged_repo = staged_repo


class DocStringParserConstants:
    """Constants used by DocStringParser for parsing docstrings."""

    pipe_prefixes: List[str] = [
        ">>> motion_adapter = ",
        ">>> adapter = ",  # if this moves, also change motion_adapter check
        ">>> controlnet = ",
        ">>> super_res_1_pipe = ",
        ">>> pipe_prior = ",
        ">>> pipe_prior_redux = ",
        ">>> pipe = ",
        ">>> pipeline = ",
        ">>> blip_diffusion_pipe = ",
        ">>> prior_pipe = ",
        ">>> gen_pipe = ",
        "pipe = ",
    ]
    repo_variables: List[str] = [
        "controlnet_model",
        "controlnet_id",
        "base_model",
        "model_id_or_path",
        "model_ckpt",
        "model_id",
        "repo_base",
        "repo",
        "motion_adapter_id",
    ]
    call_types: List[str] = [".from_pretrained(", ".from_single_file("]
    staged_call_types: List[str] = [
        ".from_pretrain(",
    ]


package_map = {
    "diffusers": ("_import_structure", "diffusers.pipelines"),
    "transformers": ("MODEL_MAPPING_NAMES", "transformers.models.auto.modeling_auto"),
}
root_path = os.path.join(os.getcwd(), "mir")
versions = read_json_file(os.path.join(root_path, "spec", "versions.json"))
template = read_json_file(os.path.join(root_path, "spec", "template.json"))
MIR_PATH_NAMED = os.path.join(root_path, "mir.json")

BREAKING_SUFFIX = r".*(?:-)(prior)$|.*(?:-)(diffusers)$|.*[_-](\d{3,4}px|-T2V$|-I2V$)"
PARAMETERS_SUFFIX = r"(\d{1,4}[KkMmBb]|[._-]\d+[\._-]\d+[Bb][._-]).*?$"
SEARCH_SUFFIX = r"\d+[._-]?\d+[BbMmKk](it)?|[._-]\d+[BbMmKk](it)?"
