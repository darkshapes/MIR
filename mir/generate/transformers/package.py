# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from typing import Callable, ModuleType
from dataclasses import dataclass, field


@dataclass
class MIRPackage:
    config: Callable
    model: Callable
    package: dict[str, str] = field(init=False, default_factory=dict[str, str])

    def __post_init__(self):
        self.package = {}
        self.model_name: str = self.model.__name__
        self.model_path: ModuleType = self.model.__module__
        if not isinstance(self.config, dict):
            self.generate_package()
            self.generate_repo()

    def generate_repo(self) -> None:
        from mir.data import MIGRATIONS

        if repo := MIGRATIONS["config"].get(self.config.__name__, {}):
            self.repo = repo
        else:
            self.repo = self.config_to_repo(self.config)

    def generate_package(self) -> None:
        """Generates package information for the MIR tag based on class.
        :param pkg: A class object (model, tokenizer, etc) to build a tag from"""
        model = f"{self.model_type}.{self.model_name}"
        self.package: dict[str, str] = {"model": model}

    def config_to_repo(self) -> str | None:
        """Extracts the repository path from the configuration class documentation.\n
        :param config_class: Configuration class to extract repository path from.
        :return: Repository path as a string if found, otherwise None."""
        import re

        from mir import NFO

        doc_check = [self.config]
        if hasattr(self.config, "forward"):
            doc_check.append(self.config.forward)  # type: ignore
        for pattern in doc_check:
            doc_string = pattern.__doc__
            matches = re.findall(r"\[([^\]]+)\]", doc_string)  # type: ignore
            if matches:
                try:
                    return next(iter(snip.strip('"').strip() for snip in matches if "/" in snip))
                except StopIteration as error_log:
                    NFO(f"ERROR >>{matches} : LOG >> {error_log}")
                    continue
