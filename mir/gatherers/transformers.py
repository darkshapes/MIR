# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import (
    MODEL_MAPPING,  # config: model map
    MODEL_MAPPING_NAMES,
    AutoModel,
)
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

AUTO_MAP = AutoModel._model_mapping
REVERSE_MAP = AUTO_MAP._reverse_config_mapping


class GatherLoop:
    def __init__(self) -> None:
        """Loops through transformers packages to harvest class data."""
        from mir.build_entry import BuildEntry
        from mir.maid import MIRDatabase

        self.db = MIRDatabase()

        build_entries = []
        for config, model in AUTO_MAP.items():  # type: ignore
            if isinstance(model, tuple):
                model: Callable = model[0]  # type: ignore
            build_entries.append(BuildEntry("model", model))
            if tokenizer := TOKENIZER_MAPPING.get(config, None):
                build_entries.append(BuildEntry("tokenizer", tokenizer))
        self.model_db = {x.attributes.model_name: x.attributes.model_parameters for x in build_entries}
