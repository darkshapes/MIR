# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Any, Callable

from chanfig import NestedDict

from mir.generate.transformers.raw_data import PrepareData


class HarvestClasses:
    def __init__(self) -> None:
        """Initializes the HarvestClasses instance with an empty list to store raw class data."""
        self.raw_data = []
        from mir.maid import MIRDatabase

        self.mir_db = MIRDatabase()
        self.find_transformers_classes()
        self.info = NestedDict({})

    def find_transformers_classes(self) -> None:
        """Finds and collects PrepareData entries for all transformer classes defined in AUTO_MAP.\n
        :return: List of PrepareData entries representing the transformer classes."""

        from mir.generate.transformers import AUTO_MAP

        model_data = []
        for pair_map in AUTO_MAP.items():
            config_class, model_class = pair_map  # type:ignore
            if isinstance(model_class, tuple):
                model_class: Callable = model_class[0]
            print(model_class)
            if config_data := self.extract_config_class_data(config_class):
                if model_data := self.extract_model_class_data(model_class):
                    if prepared_data := PrepareData(**config_data, **model_data):  # type:ignore
                        self.add_to_database(prepared_data)

    def extract_config_class_data(self, config_class: Callable) -> dict[str, str | Callable | dict[str, Any]] | None:
        """Extracts information from config classes.\n
        :param config_class: Model class or callable returning model classes.
        :return: dictionary of discovered elements"""
        from mir.data import MIGRATIONS, PARAMETERS
        from mir.generate.from_module import show_init_fields_for

        config_name = config_class.__name__
        config_params = PARAMETERS.get(config_name, {})
        repo_path = MIGRATIONS["config"].get(config_name, {})
        if not config_params:
            config_params = show_init_fields_for(config_class)
        if not repo_path:
            repo_path = self.config_to_repo(config_class)
        if not repo_path or not config_params or "inspect" in config_params or "deprecated" in config_params:
            return None
        return {
            "name": config_name,
            "config": config_class,
            "config_params": config_params,
            "repo_path": repo_path,
        }

    def extract_model_class_data(self, model_class: Callable) -> dict[str, str | Any] | None:
        """Extracts information from model classes.\n
        :param model_class: Model class or callable returning model classes.
        :return: dictionary of discovered elements"""
        from mir.generate.from_module import show_init_fields_for  # Ensure it's a tuple for consistency.

        model_data: dict[str, str | Any] = {"model": model_class}
        model_params = show_init_fields_for(model_class)
        if "inspect" in model_params or "deprecated" in model_params:
            return None
        else:
            return model_data | {
                "model_params": model_params,
            }

    def config_to_repo(self, config_class: Callable) -> str | None:
        """Extracts the repository path from the configuration class documentation.\n
        :param config_class: Configuration class to extract repository path from.
        :return: Repository path as a string if found, otherwise None."""
        import re

        from mir import NFO

        doc_check = [config_class]
        if hasattr(config_class, "forward"):
            doc_check.append(config_class.forward)  # type: ignore
        for pattern in doc_check:
            doc_string = pattern.__doc__
            repo_brackets = r"\[([^\]]+)\]"
            matches = re.findall(repo_brackets, doc_string)  # type: ignore
            if matches:
                try:
                    self.repo_path = next(iter(snip.strip('"').strip() for snip in matches if "/" in snip))
                except StopIteration as error_log:
                    NFO(f"ERROR >>{matches} : LOG >> {error_log}")
                    continue

    def add_to_database(self, prepared_data: PrepareData) -> None:
        if hasattr(prepared_data, "tokenizer"):
            token_info = NestedDict(
                {
                    "encoder": {
                        "tokenizer": {
                            prepared_data.mir_comp: {
                                "pkg": {f"{prepared_data.tokenizer.__module__}.{prepared_data.tokenizer.__name__}"},
                            },
                        },
                    },
                }
            )

        info = NestedDict(
            {
                prepared_data.mir_arch: {
                    prepared_data.mir_series: {
                        prepared_data.mir_comp: {
                            "repo": prepared_data.repo_path,
                            "pkg": {"transformers": prepared_data.model_name},
                            "tokenizer": {f"info.encoder.tokenizer.{prepared_data.mir_comp}"},
                        }
                    }
                }
            }
        )
        self.info = token_info | info
        print(f"added {prepared_data}")
