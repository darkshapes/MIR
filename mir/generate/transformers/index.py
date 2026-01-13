# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from typing import Callable

from mir import NFO, DBUQ
from mir.data import PARAMETERS
from mir.generate.from_module import import_object_named, to_domain_tag
from mir.generate.indexers import migrations
from mir.tag import tag_model_from_repo
from mir.generate.transformers import CONFIG_MAPPING, MODEL_MAPPING, TOKENIZER_MAPPING_NAMES, ClassMapEntry


def mapped_cls(model_identifier: str):
    """Get model class from identifier without calling huggingface_hub.\n
    :param model_identifier: Model identifier like "bert-base-uncased" or "gpt2"
    :return: Model class (e.g., BertModel, GPT2Model)
    """
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_MAPPING_NAMES

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


def get_repo_from_class_map(class_map: ClassMapEntry) -> str | None:
    """The name of the repository that is associated with a transformers configuration class\n
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
                NFO(f"ERROR >>{matches} : LOG >> {error_log}")
                continue
            return repo_path
    return None


def find_transformers_classes() -> list[ClassMapEntry]:
    """Eat the 🤗Transformers classes as a treat, leaving any tasty subclass class morsels neatly arranged as a dictionary.\n
    Nom.\n
    :return: Tasty mapping of subclasses to their class references"""

    model_data = []
    for config_name, config_obj in CONFIG_MAPPING.items():
        model_params = None
        if model_obj := MODEL_MAPPING.get(config_obj, None):
            if isinstance(model_obj, Callable):
                model_obj = (model_obj,)
            assert isinstance(model_obj, tuple), f"Expected model class object, got {model_obj} type {type(model_obj)}"
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


def mir_tag_from_config(class_map: ClassMapEntry, repo_path: str) -> tuple[str, str, str]:
    """Change a transformers config class into a MIR series and comp\n
    :param class_map: Transformers class information extracted from dependency
    :param repo_path: The
    """

    mir_prefix = to_domain_tag(transformers=True, **class_map.config_params)
    if not mir_prefix:
        if class_map.model_params:
            if mir_prefix := to_domain_tag(transformers=True, **class_map.model_params):
                pass
            else:
                raise ValueError(f"Unable to determine MIR prefix from {class_map, repo_path}")
        else:
            raise ValueError(f"Unrecognized model type, no tag matched {class_map.name} with {class_map.config_params} or {class_map.model_params}")
    mir_prefix = "info." + mir_prefix
    if class_map.name != "funnel":
        mir_suffix, mir_comp = tag_model_from_repo(repo_path)
    else:
        mir_suffix, mir_comp = ["funnel", "*"]
    mir_series = mir_prefix + "." + mir_suffix
    return mir_series, mir_comp, mir_suffix


def show_transformers_tasks(class_name: str | None = None, code_name: str | None = None) -> list[str]:
    """Retrieves a list of task classes associated with a specified transformer class.\n
    :param class_name: The name of the transformer class to inspect.
    :param pkg_type: The dependency for the module
    :param alt_method: Use an alternate method to return the classes
    :return: A list of task classes associated with the specified transformer."""

    task_classes = None

    if not code_name:
        class_obj: Callable = import_object_named(class_name, "transformers")
        class_module: Callable = import_object_named(*class_obj.__module__.split(".", 1)[-1:], class_obj.__module__.split(".", 1)[0])
        if class_module and class_module.__name__ != "DummyPipe":
            task_classes = getattr(class_module, "__all__")
        else:
            return None
    elif code_name:
        from httpx import HTTPStatusError

        from mir.generate.transformers.index import mapped_cls

        try:
            model_class = mapped_cls(code_name)
            if model_class is not None:
                # Convert class type to list containing the class name string
                task_classes = [model_class.__name__]
            else:
                return None
        except (OSError, HTTPStatusError) as e:
            DBUQ(f"Error mapping class {code_name}: {e}")
            return None

    return task_classes


def transformers_index():
    """Generate LLM model data for MIR index\n
    :return: Dictionary ready to be applied to MIR data fields"""

    missing_config_params = PARAMETERS

    mir_data = {}
    transformers_data: list[ClassMapEntry] = find_transformers_classes()
    for entry in transformers_data:
        repo_path = get_repo_from_class_map(entry)
        if entry.name == "bert":
            print(entry)
        if config := missing_config_params.get(entry.name, {}):
            entry.config_params = config.get("params", entry.config_params)
            repo_path = config.get("repo_path", repo_path)
        if entry.name == "bert":
            print(entry)
        if not repo_path:
            raise ValueError(f"Unable to determine repo from {entry}")
        if entry.config_params:
            mir_series, mir_comp, mir_suffix = mir_tag_from_config(entry, repo_path)
            # modalities = add_mode_types(mir_tag=[mir_series, mir_comp])

            repo_path = migrations(repo_path)
            tk_pkg = {}
            tokenizer_classes = TOKENIZER_MAPPING_NAMES.get(entry.name)
            if isinstance(tokenizer_classes, str):
                tokenizer_classes = [tokenizer_classes]
            # mode = modalities.get("mode")
            if tokenizer_classes:
                index = 0
                for tokenizer in tokenizer_classes:
                    if tokenizer:
                        tokenizer_class = import_object_named(tokenizer, "transformers")
                        tk_pkg.setdefault(index, {"transformers": f"{tokenizer_class.__module__}.{tokenizer_class.__name__}"})
                        index += 1
                if tk_pkg:
                    mir_data.get("info.encoder.tokenizer", mir_data.setdefault("info.encoder.tokenizer", {})).update(
                        {
                            mir_suffix: {
                                "pkg": tk_pkg,
                            }
                        },
                    )
            mir_data.setdefault(
                mir_series,
                {
                    mir_comp: {
                        "repo": repo_path,
                        "pkg": {
                            0: {"transformers": entry.model_name},
                        },
                        # "mode": mode,
                    },
                },
            )
    return mir_data
