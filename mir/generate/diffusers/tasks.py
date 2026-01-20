# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CollectTasks:
    model: Callable
    import_path: str
    tasks: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.model_to_tasks()

    def model_to_tasks(self) -> None:
        """Return Diffusers task pipes based on package-specific query\n
        :param class_name: To find task pipes from a Diffusers class pipe, defaults to None
        :param code_name: To find task pipes from a Transformers class pipe, defaults to None
        :return: A list of alternate class pipelines derived from the specified class"""
        from mir.generate.diffusers import SUPPORTED_TASKS_MAPPINGS, GET_TASK_CLASS

        alt_tasks = set({})
        self.internal_name = self.import_path.rsplit(".", 2)[-1]
        for task_map in SUPPORTED_TASKS_MAPPINGS:
            task_class = GET_TASK_CLASS(task_map, self.model, False)
            if task_class:
                alt_tasks.add(task_class.__name__)
            for model_code, pipe_class_obj in task_map.items():
                if self.internal_name in model_code:
                    alt_tasks.add(pipe_class_obj.__name__)

        self.tasks = [x for x in alt_tasks]
