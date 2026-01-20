# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable

from mir.generate.transformers.raw_data import PrepareData


class HarvestLoop:
    def __init__(self) -> None:
        """Initializes the HarvestClasses instance with an empty list to store raw class data."""
        from mir.maid import MIRDatabase

        self.db = MIRDatabase()

    def __call__(self) -> None:
        from mir.generate.transformers import AUTO_MAP
        from mir.generate.transformers import TOKENIZER_MAPPING

        prepared_data = {}
        for config_class, model_data in AUTO_MAP.items():
            assert isinstance(config_class, Callable)
            loop_parameters = {"model": (model_data, config_class)}
            if tokenizer := TOKENIZER_MAPPING.get(config_class, None):
                loop_parameters.setdefault("tokenizer", (tokenizer, tokenizer))  # type: ignore
            for name, (self.model, self.config) in loop_parameters.items():
                if prepare_data := self.prepare_class_data():  # type: ignore
                    prepared_data.setdefault(name, prepare_data)
        for data in prepared_data:
            pass

    def prepare_class_data(self) -> PrepareData | None:
        """Extract and collect information from model and config classes.\n
        :return: A PrepareData entry representing the transformer class."""
        from mir.data import PARAMETERS
        from mir.generate.from_module import show_init_fields_for

        config_name = self.config.__name__
        config_params = PARAMETERS.get(config_name, show_init_fields_for(self.config))
        if any(x in config_params for x in ["inspect", "deprecated"]):
            return None
        if isinstance(self.model, tuple):
            self.model_class: Callable = self.model[0]
        return PrepareData(model=self.model, **config_params)  # type: ignore
