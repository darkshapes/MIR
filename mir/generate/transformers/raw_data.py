# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PrepareData:
    """Represents a structured entry of the name of the class and its associated attributes."""

    name: str
    model: Callable
    config: Callable
    repo_path: str
    config_params: dict[str, list[str]]
    model_params: dict[str, list[str]] | None = None
    mir_arch: str = field(init=False)
    mir_series: str = field(init=False)
    mir_comp: str = field(init=False)

    def __post_init__(self) -> None:
        """Initializes the PrepareData instance by setting derived attributes."""
        from mir.generate.transformers import REVERSE_MAP, TOKENIZER_MAPPING

        self.model_name: str = self.model.__name__.split(".")[-1]
        if tokenizer := TOKENIZER_MAPPING.get(self.config, None):
            self.tokenizer = tokenizer
            self.tokenizer_pkg: dict[str, str] | None = {"transformers": f"{self.tokenizer.__module__}.{self.tokenizer.__name__}"}
        if internal_name := REVERSE_MAP.get(self.config):
            self.internal_name = internal_name
        self.model_to_tasks()
        self.mir_tag_from_config()

    def model_to_tasks(self) -> None:
        """Transform a single model class into derivative classes for specific tasks.\n
        :return: A list of task classes associated with the model."""
        from pathlib import Path
        from importlib import import_module

        import_path = Path(self.model.__module__).stem
        parent_module = import_module(import_path)

        if hasattr(parent_module, "__all__") and parent_module.__name__ != "DummyPipe":
            self.task_classes = parent_module.__all__
        else:
            self.task_classes = [self.model.__name__]

    def mir_tag_from_config(self) -> None:
        """Generates MIR series and component tags based on the configuration class.\n
        :return: Tuple containing MIR series, component, and suffix tags."""

        from mir.generate.from_module import to_domain_tag
        from mir.tag import tag_model_from_repo

        mir_prefix = to_domain_tag(transformers=True, **self.config_params)
        if not mir_prefix:
            if self.model_params:
                if mir_prefix := to_domain_tag(transformers=True, **self.model_params):
                    pass
                raise ValueError(f"Unable to determine MIR prefix from {self}")
            else:
                raise ValueError(f"Unrecognized model type, no tag matched {self.name} with {self.config_params} or {self.model_params}")
        self.mir_arch = mir_prefix
        self.mir_series, self.mir_comp = tag_model_from_repo(self.repo_path)
