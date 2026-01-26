# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable
from dataclasses import dataclass, field


@dataclass
class ModelAttributes:
    """Represents a structured entry of the class and its associated attributes.\n

    model_type: The kind of model.
    model: The model function.
    model_parameters: Dictionary mapping configuration parameter fields.
    model_name: Name of the model function.
    config: The config function for the model.
    library: Name of the library containing the model.
    import_path: Import path of the model module (excluding the package name)."""

    model: Callable
    model_type: str
    model_parameters: dict[str, list[str]] | None = None

    model_name: str = field(init=False)
    library: str = field(init=False)
    import_path: str = field(init=False)

    def __post_init__(self) -> None:
        """Initializes the instance by setting derived attributes."""
        self.model_name: str = self.model.__name__
        self.import_path = self.model.__module__.rsplit(".", 1)[0]
        self.library = self.import_path.split(".")[0]
        if not hasattr(self, "config") and any(x in self.model_type for x in ["tokenizer", "prior_tokenizer"]):
            self.config = self.model
        elif not hasattr(self, "config") and self.library == "transformers" and "model" in self.model_type:
            from mir.gatherers.transformers import AUTO_MAP

            config: dict = {model: config for config, model in AUTO_MAP.items() if model == self.model}
            self.config = config.get(self.model, None)  # type:ignore
        if getattr(self, "config", None) and self.library == "transformers":
            from mir.data import PARAMETERS
            from mir.lookups import show_init_fields_for

            config_name = self.config.__name__
            config_parameters = PARAMETERS.get(config_name, show_init_fields_for(self.config))
            if not any(x in config_parameters for x in ["inspect", "deprecated"]):
                self.config = self.config
                self.model_parameters = config_parameters
            else:
                self.model_parameters = None
