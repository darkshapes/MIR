# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from dataclasses import dataclass, field
from typing import Callable, get_type_hints


@dataclass
class DPrepareData:
    doc_string: str
    model: Callable
    model_path: str
    library: str
    model_name: str
    model_params: dict[str, list[str]] = field(init=True, default_factory=lambda: {"": [""]})
    repo_path: str = field(init=False, default_factory=str)
    staged_repo: str | None = field(init=False, default_factory=str)
    tasks: list[str] = field(init=False, default_factory=lambda: [""])
    name: str = field(init=False, default_factory=str)

    def __post_init__(self) -> None:
        from mir.data import MIGRATIONS
        from mir.generate.diffusers.doc_parse import DocStringParser

        doc_parser = DocStringParser(doc_string=self.doc_string, model=self.model, model_path=self.model_path)
        doc_parser.parse()
        if repo_path := MIGRATIONS["migrated_pipes"].get(self.model.__name__, False):
            self.repo_path = repo_path
        else:
            if repo_path := doc_parser.pipe_repo:
                self.repo_path = repo_path
            if staged_repo := doc_parser.staged_repo:
                self.staged_repo = staged_repo
        self.show_diffusers_tasks()
        for name, model in self.model_params.items():
            setattr(self, name, model)
            print(name, model)

    def show_diffusers_tasks(self) -> None:
        """Return Diffusers task pipes based on package-specific query\n
        :param class_name: To find task pipes from a Diffusers class pipe, defaults to None
        :param code_name: To find task pipes from a Transformers class pipe, defaults to None
        :return: A list of alternate class pipelines derived from the specified class"""
        from mir.generate.diffusers import SUPPORTED_TASKS_MAPPINGS, GET_TASK_CLASS

        alt_tasks = set({})
        self.internal_name = self.model_path.rsplit(".", 2)[-1]
        for task_map in SUPPORTED_TASKS_MAPPINGS:
            task_class = GET_TASK_CLASS(task_map, self.model, False)
            if task_class:
                alt_tasks.add(task_class.__name__)
            for model_code, pipe_class_obj in task_map.items():
                if self.internal_name in model_code:
                    alt_tasks.add(pipe_class_obj.__name__)

        self.tasks = [x for x in alt_tasks]
