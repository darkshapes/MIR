# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import dataclass
from mir.model import ModelAttributes
from mir.data import MIGRATIONS


@dataclass
class MIRPackage:
    attributes: ModelAttributes

    def __post_init__(self):
        self.package = {}
        if self.attributes.model_type == "model":
            if self.attributes.library == "transformers":
                self.package_transformers()
            elif self.attributes.library == "diffusers":
                self.package_diffusers()
        model = f"{self.attributes.import_path}.{self.attributes.model_name}"
        self.package: dict[str, str] = {"model": model}

    def package_transformers(self) -> None:
        """Generates package information for the MIR tag based on class."""

        if hasattr(self.attributes, "config"):
            config_name = self.attributes.config.__name__
            if repo := MIGRATIONS["config"].get(config_name, {}):
                self.repo = repo
            else:
                self.repo_from_config()
            self.tasks_from_model()

    def package_diffusers(self) -> None:
        """Generates package information for the MIR tag based on class."""

        if repo := MIGRATIONS["migrated_pipes"].get(self.attributes.model_name, False):
            self.repo = repo
        elif doc_string := getattr(self.attributes.import_path, "EXAMPLE_DOC_STRING", None) and not any(x in self.attributes.model_type for x in ["tokenizer", "scheduler"]):
            self.repo_from_doc_string(doc_string=doc_string)  # type: ignore
        self.tasks_from_internal_name()

    def repo_from_config(self) -> None:
        """Extracts the repository path from the configuration class documentation.\n
        :param config_class: Configuration class to extract repository path from.
        :return: Repository path as a string if found, otherwise None."""
        import re

        from mir import NFO

        doc_check = [self.attributes.config]
        if hasattr(self.attributes.config, "forward"):
            doc_check.append(self.config.forward)  # type: ignore
        for pattern in doc_check:
            doc_string = pattern.__doc__
            matches = re.findall(r"\[([^\]]+)\]", doc_string)  # type: ignore
            if matches:
                try:
                    self.repo = next(iter(snip.strip('"').strip() for snip in matches if "/" in snip))
                except StopIteration as error_log:
                    NFO(f"ERROR >>{matches} : LOG >> {error_log}")
                    continue

    def repo_from_doc_string(self, doc_string: str) -> None:
        from mir.generate.diffusers.doc_parse import DocStringParser

        doc_parser = DocStringParser(
            doc_string=doc_string,
            model=self.attributes.model,
            model_path=self.attributes.import_path,
        )
        doc_parser.parse()
        if repo_path := doc_parser.pipe_repo:
            self.repo = repo_path
        if staged_repo := doc_parser.staged_repo:
            self.staged_repo = staged_repo

    def tasks_from_internal_name(self) -> None:
        """Return Diffusers task pipes based on package-specific query\n
        :param class_name: To find task pipes from a Diffusers class pipe, defaults to None
        :param code_name: To find task pipes from a Transformers class pipe, defaults to None
        :return: A list of alternate class pipelines derived from the specified class"""
        from mir.generate.diffusers import SUPPORTED_TASKS_MAPPINGS, GET_TASK_CLASS

        alt_tasks = set({})
        self.internal_name = self.attributes.import_path.rsplit(".", 2)[-1]
        for task_map in SUPPORTED_TASKS_MAPPINGS:
            task_class = GET_TASK_CLASS(task_map, self.attributes.model, False)
            if task_class:
                alt_tasks.add(task_class.__name__)
            for model_code, pipe_class_obj in task_map.items():
                if self.internal_name in model_code:
                    alt_tasks.add(pipe_class_obj.__name__)
        if alt_tasks:
            self.tasks = [x for x in alt_tasks]

    def tasks_from_model(self) -> None:
        """Transform a single model class into derivative classes for specific tasks.\n
        :return: A list of task classes associated with the model."""
        from importlib import import_module

        model_name = self.attributes.model_name

        parent_module = import_module(self.attributes.import_path)
        self.tasks = []
        if hasattr(parent_module, "__all__") and parent_module.__name__ != "DummyPipe":
            for module in parent_module.__all__:
                if (module.lower() != module) and (module != model_name) and (module != self.attributes.config.__name__):
                    self.tasks.append(module)
        else:
            self.tasks = [model_name]
