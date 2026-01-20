# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CollectTasks:
    model: Callable
    import_path: str
    config: Callable
    tasks: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.model_to_tasks()

    def model_to_tasks(self) -> None:
        """Transform a single model class into derivative classes for specific tasks.\n
        :return: A list of task classes associated with the model."""
        from importlib import import_module

        model_name = self.model.__name__

        parent_module = import_module(self.import_path)
        self.tasks = []
        if hasattr(parent_module, "__all__") and parent_module.__name__ != "DummyPipe":
            for module in parent_module.__all__:
                if (module.lower() != module) and (module != model_name) and (module != self.config.__name__):
                    self.tasks.append(module)
        else:
            self.tasks = [model_name]
