# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from dataclasses import dataclass, field
from typing import Callable, get_type_hints


@dataclass
class DPrepareData:
    name: str
    doc_string: str
    model: Callable
    model_path: str
    repo_path: str = field(init=False, default_factory=str)
    model_name: str = field(init=False, default_factory=str)
    staged_repo: str | None = field(init=False, default_factory=str | None)
    tasks: list[str] = field(init=False, default_factory=lambda: [""])

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __post_init__(self) -> None:
        from mir.data import MIGRATIONS
        from mir.generate.diffusers.doc_parse import DocStringParser
        from mir.generate.from_module import show_init_fields_for

        self.model_name = self.model.__name__
        self.library = self.model.__module__.split(".", 1)[0]
        self.model_params = show_init_fields_for(self.model, "diffusers")
        self.type_params = get_type_hints(self.model.__init__)
        doc_parser = DocStringParser(self.doc_string, self.model)
        if repo_path := MIGRATIONS["migrated_pipes"].get(self.model.__name__, False):
            self.repo_path = repo_path
        else:
            if repo_path := doc_parser.pipe_repo:
                self.repo_path = repo_path
            if staged_repo := doc_parser.staged_repo:
                self.staged_repo = staged_repo

    def show_diffusers_tasks(self) -> list[str]:
        """Return Diffusers task pipes based on package-specific query\n
        :param class_name: To find task pipes from a Diffusers class pipe, defaults to None
        :param code_name: To find task pipes from a Transformers class pipe, defaults to None
        :return: A list of alternate class pipelines derived from the specified class"""
        from mir.generate.diffusers import SUPPORTED_TASKS_MAPPINGS, GET_TASK_CLASS

        alt_tasks = set()
        internal_name = self.model_path.rsplit(".", 2)[-2]
        for task_map in SUPPORTED_TASKS_MAPPINGS:
            task_class = GET_TASK_CLASS(task_map, self.model, False)
            if task_class:
                alt_tasks.add(task_class.__name__)
            for model_code, pipe_class_obj in task_map.items():
                if internal_name in model_code:
                    alt_tasks.add(pipe_class_obj.__name__)

        return list(alt_tasks)
