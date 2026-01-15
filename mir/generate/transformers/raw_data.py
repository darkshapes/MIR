# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class PrepareData:
    """Represents a structured entry of the name of the class and its associated attributes."""

    name: str
    model: Callable
    config: type
    repo_path: str
    config_params: dict[str, list[str]]
    model_params: dict[str, list[str]] | None = field(init=True, default_factory=lambda: {"": [""]})
    tasks: list[str] = field(init=False, default_factory=lambda: [""])

    def __post_init__(self) -> None:
        """Initializes the PrepareData instance by setting derived attributes."""
        from mir.generate.transformers import REVERSE_MAP, TOKENIZER_MAPPING

        self.model_name: str = self.model.__name__.split(".")[-1]
        if tokenizer := TOKENIZER_MAPPING.get(self.config, None):
            self.tokenizer: tuple[type[Any] | None, type[Any] | None] = tokenizer
        if internal_name := REVERSE_MAP.get(self.config):
            self.internal_name = internal_name
        self.model_to_tasks()

    def model_to_tasks(self) -> None:
        """Transform a single model class into derivative classes for specific tasks.\n
        :return: A list of task classes associated with the model."""
        from pathlib import Path
        from importlib import import_module

        import_path = Path(self.model.__module__).stem
        parent_module = import_module(import_path)
        self.tasks = []
        if hasattr(parent_module, "__all__") and parent_module.__name__ != "DummyPipe":
            for module in parent_module.__all__:
                if (module.lower() != module) and (module != self.model_name) and (module != self.config.__name__):
                    self.tasks.append(module)
        else:
            self.tasks = [self.model.__name__]
