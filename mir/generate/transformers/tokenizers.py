# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import re
from importlib import import_module

from mir.generate.transformers import TOKENIZER_MAPPING
from mir.maid import MIRDatabase
from mir.spec import mir_entry


def tag_tokenizers(config_class: Callable):
    tokenizer_class = TOKENIZER_MAPPING[config_class]  # type: ignore
    if tokenizer_class:
        { "pkg":{"transformers": f"{tokenizer_class.__module__}.{tokenizer_class.__name__}"})
        if tk_pkg:
            mir_data.get("info.encoder.tokenizer", mir_data.setdefault("info.encoder.tokenizer", {})).update(
                {
                    mir_suffix: {
                        "pkg": tk_pkg,
                    }
                },
            )
    return tokenizer_class
