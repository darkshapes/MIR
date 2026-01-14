# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import (
    MODEL_MAPPING,  # config: model map
    MODEL_MAPPING_NAMES,
    AutoModel,
)
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

from mir.generate.from_module import show_init_fields_for

AUTO_MAP = AutoModel._model_mapping
REVERSE_MAP = AUTO_MAP._reverse_config_mapping
