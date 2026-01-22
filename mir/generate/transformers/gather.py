# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable


class GatherLoop:
    def __init__(self) -> None:
        """Loops through transformers packages to harvest class data."""
        from mir.generate.transformers import AUTO_MAP
        from mir.generate.transformers import TOKENIZER_MAPPING
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
        print([x.attributes for x in build_entries])  # type: ignore
