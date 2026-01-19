# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Any, Callable
from dataclasses import dataclass, field
from mir.generate.diffusers.raw_data import DPrepareData
from mir.generate.transformers.raw_data import PrepareData
from mir.tag import MIRTag


@dataclass
class MIRPackage:
    data: Callable | str | dict[str, str]
    package: dict[str, str] = field(init=False, default_factory=dict[str, str])

    def __init__(self, data: Callable | str | dict[str, str] | dict[str, Any]):
        self.package = {}
        self.data = data
        if not isinstance(self.data, dict):
            self.generate_package()
        else:
            self.add_framework(self.data)

    def generate_package(self) -> None:
        """Generates package information for the MIR tag based on class.
        :param pkg: A class object (model, tokenizer, etc) to build a tag from"""
        self.domain = "ops"
        model = f"{self.data.__module__}.{self.data.__name__}"
        self.package: dict[str, str] = {"model": model}

    def add_framework(self, framework_data) -> None:
        self.domain = "info"
        self.package = framework_data


class MIRNesting:
    """Build tag components from the extracted data\n
    :param mir_tag: An instance of MIR tag with the necessary information
    :param prepared_data: Instance of PrepareData to attribute the final information
    :returns: The final, assembled MIR tag"""

    loops: list[str]
    framework_data: dict[str, str | dict[str, Any]] = {}
    repo: str | None = field(default_factory=str | None)
    framework: dict[str, str] = field(init=False)
    tokenizer: str | None = field(default_factory=str)

    def __init__(self, mir_tag: MIRTag, prepared_data: PrepareData | DPrepareData) -> None:
        """\nInitialize the framework with MIR tag and prepared data.\n
        :param mir_tag : The MIR tag instance.
        :param prepared_data : The prepared data for processing."""
        self.mir_tag = mir_tag

        self.prepared_data = prepared_data
        self.loops = []
        self.framework_data = {}

    def __call__(self, packages: dict[str, MIRPackage]) -> None:
        """Common routine for handling a package: store tag data, nest the package,
        and record the name of the newly-created attribute.\n
        :param name: Identification string to store data underneath
        :param mir_package: An instance of MIRPackage with the requisite data"""

        for name, mir_package in packages.items():
            is_framework = name == "framework"
            is_model = name == "model"

            if is_framework:
                package_data = {self.prepared_data.library: mir_package.package}
                tag_data = f"{mir_package.domain}.{self.mir_tag.arch}.{self.mir_tag.series}"
                if comp := getattr(self.mir_tag, "comp", None):
                    tag_data += comp
                self.framework_data.setdefault("repo", self.prepared_data.repo_path)
            elif is_model:
                package_data = {self.prepared_data.library: mir_package.package}
                if hasattr(self.prepared_data, "tasks") and self.prepared_data.tasks:
                    package_data[self.prepared_data.library].setdefault("tasks", self.prepared_data.tasks)
                tag_data = f"{mir_package.domain}.{self.mir_tag.arch}.{self.mir_tag.series}"
                if comp := getattr(self.mir_tag, "comp", None):
                    tag_data += comp
                self.framework_data.setdefault(name, tag_data)
            elif is_tokenizer:  # tokenizer case
                package_data = {self.prepared_data.library: mir_package.package}
                tag_data = f"{mir_package.domain}.encoder.tokenizer.{self.mir_tag.series}"
                self.framework_data.setdefault(name, tag_data)

            self.nest_data(name=name, tag_data=tag_data, package_data=package_data)
            self.loops.append(name)

    def nest_data(self, name: str, tag_data: str, package_data: dict) -> None:
        """Nest data into a hierarchical attribute structure.\n
        :param name: Attribute name to store the nested data
        :param tag_data: Dotted path string for nesting
        :param package_data: Data to be stored in the nested structure"""

        from chanfig import NestedDict

        tag_parts = tuple(x for x in tag_data.split("."))

        if len(tag_parts) == 4:
            domain, arch, series, comp = tag_parts
            nest = NestedDict({f"{domain}.{arch}.{series}": {comp: ""}})
            nest[domain][arch][series][comp] = package_data
        else:
            domain, arch, series = tag_parts
            nest = NestedDict({f"{domain}.{arch}": {series: ""}})
            nest[domain][arch][series] = package_data

        setattr(self, name, nest)
