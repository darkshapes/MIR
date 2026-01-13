# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from dataclasses import dataclass, field
from typing import Callable

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import (
    MODEL_MAPPING,  # config: model map
    MODEL_MAPPING_NAMES,
)
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES

from mir.generate.from_module import show_init_fields_for


@dataclass
class ClassMapEntry:
    """Represents a structured entry of the name of the class and its associated attributes."""

    name: str
    model_name: str
    model: Callable
    config: Callable
    config_params: dict[str, list[str]] = field(init=False, default_factory=lambda: {})
    model_params: dict[str, list[str]] | None = None

    def __post_init__(self):
        if self.model:
            self.model_params = show_init_fields_for(self.model)
        if self.config:
            self.config_params = show_init_fields_for(self.config)
