# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable, Generator

import diffusers
from mir.config.constants import ClassMapEntry, DocStringEntry, extract_init_parameters
from mir.config.conversion import retrieve_diffusers_docstrings


#     if code_name and "__" not in code_name:
#         tasks = TaskAnalyzer.show_transformers_tasks(code_name=code_name)
#         if tasks and isinstance(tasks, list):  # Ensure tasks is a list
#             task_pipe = next(iter(tasks))
#             if isinstance(task_pipe, tuple):
#                 task_pipe = task_pipe[0]
#             if task_pipe not in exclude_list:
#                 model_class = getattr(__import__("transformers"), task_pipe)  # this is done to get the path to the config
#                 model_data = extract_init_params(model_class)
#                 if model_data and ("inspect" not in model_data["config"]) and ("deprecated" not in list(model_data["config"])):
#                     transformer_data.setdefault(model_class, model_data)
#                 else:
#                     model_data = None
#         # Reset task_pipe if tasks was None or not a list
#         if not tasks or not isinstance(tasks, list):
#             task_pipe = None

#     if not model_data and code_name not in second_exclude_list:  # second attempt
#         if code_name == "donut":
#             code_name = "donut-swin"
#         if not task_pipe and code_name and MODEL_MAPPING_NAMES.get(code_name.replace("_", "-")):
#             model_class = getattr(__import__("transformers"), MODEL_MAPPING_NAMES[code_name.replace("_", "-")], None)
#         elif task_pipe:
#             model_class = getattr(__import__("transformers"), task_pipe)
#         config_class = CONFIG_MAPPING_NAMES.get(code_name.replace("_", "-"))
#         if not config_class:
#             config_class = CONFIG_MAPPING_NAMES.get(code_name.replace("-", "_"))
#         if config_class:
#             config_class_obj = getattr(__import__("transformers"), config_class)
#             model_data = {"config": str(config_class_obj.__module__ + "." + config_class_obj.__name__).split(".")}
#             if model_data and ("inspect" not in model_data) and ("deprecated" not in model_data) and model_class:
#                 transformer_data.setdefault(model_class, model_data)
# return transformer_data


def map_transformers_classes() -> list[ClassMapEntry]:
    """Eat the 🤗Transformers classes as a treat, leaving any tasty subclass class morsels neatly arranged as a dictionary.\n
    Nom.
    :return: Tasty mapping of subclasses to their class references"""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_MAPPING  # config: model map

    model_data = []
    for config_name, config_obj in CONFIG_MAPPING.items():
        model_params = None
        if model_obj := MODEL_MAPPING.get(config_obj, None):
            if isinstance(model_obj, Callable):
                model_obj = (model_obj,)
            assert isinstance(model_obj, tuple)
            for model_class in model_obj:
                if model_params and ("inspect" not in model_params["config"]) and ("deprecated" not in list(model_params["config"])):
                    pass
                else:
                    model_params = None
                model_name = model_class.__name__
                model_data.append(
                    ClassMapEntry(
                        name=config_name,
                        model_name=model_name.split(".")[-1],
                        model=model_class,  # type: ignore
                        config=config_obj,
                    ),
                )
    return model_data


def find_diffusers_docstrings() -> Generator[list[DocStringEntry]]:
    """Pull down docstrings from 🤗Diffusers pipelines, minimizing internet requests\n
    :return: Docstrings for common diffusers models"""
    import os

    from diffusers.pipelines import _import_structure

    from mir.config.json_io import read_json_file

    project_root = os.path.dirname(os.path.dirname(__file__))
    pattern_file = os.path.join(project_root, "spec", "docstring_patterns.json")
    docstring_patterns = read_json_file(pattern_file)
    exclusion_list = docstring_patterns["exclusion_list"]
    uncommon_naming = docstring_patterns["uncommon_naming"]
    for pipe_name in _import_structure.keys():
        if pipe_name not in exclusion_list:
            file_specific = uncommon_naming.get(pipe_name, pipe_name)
            if import_name := getattr(diffusers.pipelines, str(pipe_name)):
                file_names = list(getattr(import_name, "_import_structure", {}).keys()) or [f"pipeline_{file_specific}"]
                yield list(retrieve_diffusers_docstrings(pipe_name, file_names))
            else:
                continue
