### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

from typing import Dict, List, Union, Callable, Optional, Generator, Iterator, Tuple
import pkgutil
import diffusers.pipelines
import sys

nfo = sys.stderr.write


def make_callable(module_name: str, pkg_name_or_abs_path: str) -> Optional[Callable]:
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


def show_tasks_for(code_name: str, class_name: Optional[str] = None) -> List[str]:
    """Return Diffusers/Transformers task pipes based on package-specific query\n
    :param class_name: To find task pipes from a Diffusers class pipe, defaults to None
    :param code_name: To find task pipes from a Transformers class pipe, defaults to None
    :return: A list of alternate class pipelines derived from the specified class"""

    if class_name:
        from diffusers.pipelines.auto_pipeline import SUPPORTED_TASKS_MAPPINGS, _get_task_class

        alt_tasks = []
        for task_map in SUPPORTED_TASKS_MAPPINGS:
            task_class = _get_task_class(task_map, class_name, False)
            if task_class:
                alt_tasks.append(task_class.__name__)
            for model_code, pipe_class_obj in task_map.items():
                if code_name in model_code:
                    alt_tasks.append(pipe_class_obj.__name__)

    elif code_name:
        from transformers.utils.fx import _generate_supported_model_class_names

        alt_tasks = _generate_supported_model_class_names(code_name)
    return alt_tasks


def root_class(module: Union[Callable, str], pkg_name: Optional[str] = None) -> Dict[str, List[str]]:
    """Pick apart a Diffusers or Transformers pipeline class and find its constituent parts\n
    :param module: Origin pipeline as a class or as a string
    :param library: name of a library to import the class from, only if a string is provided
    :return: Dictionary of sub-classes from the `module`"""

    import inspect

    if pkg_name and isinstance(module, str):
        module = make_callable(module, pkg_name)
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


def stock_llm_data() -> Dict[str, List[str]]:
    """Eat the 🤗Transformers classes as a treat, leaving any tasty subclass class morsels neatly arranged as a dictionary.\n
    Nom.
    :return: _description_"""

    transformer_data = {}
    exclude_list = ["DistilBertModel", "SeamlessM4TModel", "SeamlessM4Tv2Model"]
    import os

    import transformers
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

    model_names = list(dict(MODEL_MAPPING_NAMES).keys())
    folder_data = {*model_names}
    models_folder = os.path.join(os.path.dirname(transformers.__file__), "models")
    folder_data = folder_data.union(os.listdir(models_folder))

    for code_name in folder_data:
        if code_name and "__" not in code_name:
            tasks = show_tasks_for(code_name=code_name)
            if tasks:
                task_pipe = next(iter(tasks))
                if isinstance(task_pipe, tuple):
                    task_pipe = task_pipe[0]
                if task_pipe not in exclude_list:
                    model_class = getattr(__import__("transformers"), task_pipe)
                    model_data = root_class(model_class)
                    if model_data and "inspect" not in model_data["config"] and "deprecated" not in model_data["config"]:
                        transformer_data.setdefault(model_class, model_data)
    return transformer_data


def process_with_folder_path(pkg_name: str, folder_path: bool) -> Iterator[Tuple[str, str, str]]:
    """Processes package folder paths to yield example doc strings if available.\n
    :param pkg_name: The name of the package under diffusers.pipelines.
    :param file_specific: A flag indicating whether processing is specific to certain files.
    :yield: A tuple containing (pkg_name, file_name, EXAMPLE_DOC_STRING) if found.
    """
    from importlib import import_module

    file_names = list(getattr(folder_path, "_import_structure").keys())
    for file_name in file_names:
        try:
            pkg_path = f"diffusers.pipelines.{pkg_name}.{file_name}"
            pipe_file = make_callable(file_name, pkg_path)
        except ModuleNotFoundError:
            nfo(f"Module Not Found for {pkg_name}")
            pipe_file = None

        try:
            if pipe_file and hasattr(pipe_file, "EXAMPLE_DOC_STRING"):
                yield (pkg_name, file_name, pipe_file.EXAMPLE_DOC_STRING)
            else:
                pipe_file = import_module(pkg_path)
        except AttributeError:
            nfo(f"Doc String Not Found for {pipe_file} {pkg_name}")


def process_with_file_name(pkg_name: str, file_specific: bool) -> Iterator[Tuple[str, str, str]]:
    """Processes package using file name to yield example doc strings if available.\n
    :param pkg_name: The name of the package under diffusers.pipelines.
    :param file_specific: A flag indicating whether processing is specific to certain files.
    :yield: A tuple containing (pkg_name, file_name, EXAMPLE_DOC_STRING) if found.
    """
    from importlib import import_module

    file_name = f"pipeline_{file_specific}"
    try:
        pkg_path = f"diffusers.pipelines.{pkg_name}"
        pipe_file = make_callable(file_name, pkg_path)
    except ModuleNotFoundError:
        nfo(f"Module Not Found for {pkg_name}")
        pipe_file = None
    try:
        if pipe_file and hasattr(pipe_file, "EXAMPLE_DOC_STRING"):
            yield (pkg_name, file_name, pipe_file.EXAMPLE_DOC_STRING)
        else:
            pipe_file = import_module(pkg_path)
    except AttributeError:
        nfo(f"Doc String Not Found for {pipe_file} {pkg_name}")


# Refactored main loop


def cut_docs() -> Generator:
    """Draw down docstrings from 🤗Diffusers library, minimizing internet requests\n
    :return: Docstrings for common diffusers models"""

    non_standard = {
        "cogvideo": "cogvideox",
        "cogview3": "cogview3plus",
        "deepfloyd_if": "if",
        "cosmos": "cosmos2_text2image",  # search folder for all files containing 'EXAMPLE DOC STRING'
        "visualcloze": "visualcloze_generation",
    }

    exclusion_list = [  # no doc string or other issues. all can be be gathered by other means
        "autopipeline",  #
        "diffusionpipeline",  #
        "pag",  # not model based
        "stable_diffusion_attend_and_excite",
        "stable_diffusion_sag",  #
        "t2i_adapter",
        "ledits_pp",  # "leditspp_stable_diffusion",
        "latent_consistency_models",  # "latent_consistency_text2img",
        "unclip",
        # these are uncommon afaik
        "dance_diffusion",  # no doc_string
        "dit",
        "ddim",
        "ddpm",
        "deprecated",
        "latent_diffusion",  # no doc_string
        "marigold",  # specific processing routines
        "omnigen",  # tries to import torchvision
        "paint_by_example",  # no docstring
        "pia",  # lora adapter
        "semantic_stable_diffusion",  # no_docstring
        "stable_diffusion_diffedit",
        "stable_diffusion_k_diffusion",  # tries to import k_diffusion
        "stable_diffusion_panorama",
        "stable_diffusion_safe",  # impossible
        "text_to_video_synthesis",
        "unidiffuser",
    ]

    for _, pkg_name, is_pkg in pkgutil.iter_modules(diffusers.pipelines.__path__):
        if is_pkg and pkg_name not in exclusion_list:
            file_specific = non_standard.get(pkg_name, pkg_name)
            folder_name = getattr(diffusers.pipelines, pkg_name)
            if hasattr(folder_name, "_import_structure"):
                yield from process_with_folder_path(pkg_name, folder_name)
            else:
                yield from process_with_file_name(pkg_name, file_specific)


def class_parent(code_name: str, pkg_name: str) -> Optional[List[str]]:
    """Retrieve the folder path within. Only returns if it is a valid path in the system\n
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
