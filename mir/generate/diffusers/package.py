# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from types import ModuleType
from typing import Callable
from dataclasses import dataclass, field


@dataclass
class MIRPackage:
    model_type: str
    model: Callable | str | dict[str, str]
    model_path: ModuleType
    package: dict[str, str] = field(init=False, default_factory=dict[str, str])

    def __post_init__(self):
        self.package = {}
        self.model_name: str = self.model.__name__
        self.model_path: ModuleType = self.model.__module__
        if not isinstance(self.data, dict):
            self.generate_package()
            self.generate_repo()

    def generate_repo(self):
        from mir.data import MIGRATIONS

        if self.model_type in ["unet", "transformer"] and (doc_string := getattr(self.model_path, "EXAMPLE_DOC_STRING", None)):
            if repo := MIGRATIONS["migrated_pipes"].get(self.model_name, False):
                self.repo = repo
            elif self.model_type not in ["scheduler", "vae", "tokenizer"]:
                self.process_doc_string(doc_string=doc_string)

    def generate_package(self) -> None:
        """Generates package information for the MIR tag based on class.
        :param pkg: A class object (model, tokenizer, etc) to build a tag from"""
        model = f"{self.model_path}.{self.model_name}"
        self.package: dict[str, str] = {"model": model}

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
            matches = re.findall(r"\[([^\]]+)\]", doc_string)  # type: ignore
            if matches:
                try:
                    return next(iter(snip.strip('"').strip() for snip in matches if "/" in snip))
                except StopIteration as error_log:
                    NFO(f"ERROR >>{matches} : LOG >> {error_log}")
                    continue

    def process_doc_string(self, doc_string: str) -> None:
        from mir.generate.diffusers.doc_parse import DocStringParser

        doc_parser = DocStringParser(doc_string=doc_string, model=self.model, model_path=self.model_path)
        doc_parser.parse()
        if repo_path := doc_parser.pipe_repo:
            self.repo_path = repo_path
        if staged_repo := doc_parser.staged_repo:
            self.staged_repo = staged_repo
