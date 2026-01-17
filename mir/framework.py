# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Any, Callable
from dataclasses import dataclass, field
from mir.generate.transformers.raw_data import PrepareData
from mir.tag import MIRTag


@dataclass
class MIRPackage:
    data: Callable | str | dict[str, str]
    library: str = field(init=False, default_factory=str)
    package: dict[str, dict[str, str]] = field(init=False, default_factory=dict[str, dict[str, str]])
    framework: dict[str, dict[str, str]] = field(init=False, default_factory=dict[str, dict[str, str]])

    def __init__(self):
        pass

    def __call__(self, data: Callable | str | dict[str, str]):
        self.data = data
        if isinstance(self.data, Callable):
            self.generate_package()

    def generate_package(self) -> None:
        """Generates package information for the MIR tag based on class.
        :param pkg: A class object (model, tokenizer, etc) to build a tag from"""
        self.domain = "ops"
        module_path = self.data.__module__
        self.library = module_path.split(".")[0]
        self.package: dict[str, dict[str, str]] = {self.library: {"model": f"{module_path}.{self.data.__name__}"}}

    def add_framework(self, framework_data) -> None:
        self.domain = "info"
        self.framework = {self.library: framework_data}


class MIRNesting:
    """Build tag components from the extracted data\n
    :param mir_tag: An instance of MIR tag with the necessary information
    :param name: Identification string to store data underneath
    :param mir_package: Instance of MIRPackage to store inside the nested dict
    :param prepared_data: Instance of PrepareData to attribute the final information
    :returns: The final, assembled MIR tag"""

    loops: list[str]
    framework_data: dict[str, str | dict[str, Any]] = {}
    repo: str | None = field(default_factory=str | None)
    framework: dict[str, str] = field(init=False)
    tokenizer: str | None = field(default_factory=str)

    def __init__(self, mir_tag: MIRTag) -> None:
        self.mir_tag = mir_tag
        self.loops = []
        self.framework_data = {}

    def __call__(self, mir_package: MIRPackage, prepared_data: PrepareData | None = None):
        if hasattr(mir_package, "library"):
            self.library = mir_package.library
        if prepared_data:
            self.framework_data.setdefault("repo", prepared_data.repo_path)
        if hasattr(mir_package, "tokenizer"):
            name = "tokenizer"
            self.package = mir_package.package
            self.nest_data(
                name=name,
                domain=mir_package.domain,
                arch="encoder",
                series="tokenizer",
                comp=self.mir_tag.series,
            )
            self.framework_data.setdefault("tokenizer", f"{mir_package.domain}.encoder.tokenizer.{self.mir_tag.series}")
        else:
            data = f"{mir_package.domain}.{self.mir_tag.arch}.{self.mir_tag.series}"
            if comp := getattr(self.mir_tag, "comp", None):
                self.framework_data.setdefault("model", data + comp)
            else:
                self.framework_data.setdefault("model", data)

            if hasattr(mir_package, "framework"):
                name = "framework"
                self.package = mir_package.framework
            else:
                name = "model"
                self.package = mir_package.package
                if hasattr(prepared_data, "tasks") and prepared_data.tasks:
                    self.package[mir_package.library].setdefault("tasks", prepared_data.tasks)
            self.nest_data(
                name=name,
                domain=mir_package.domain,
                arch=self.mir_tag.arch,
                series=self.mir_tag.series,
                comp=comp,
            )
        self.loops.append(name)

    def nest_data(self, name: str, domain: str, arch: str, series: str, comp: str | None = None) -> None:
        from chanfig import NestedDict

        if comp:
            nest = NestedDict({f"{domain}.{arch}.{series}": {comp: ""}})
            nest[domain][arch][series] = self.package
        else:
            nest = NestedDict({f"{domain}.{arch}": {series: ""}})
            nest[domain][arch][series] = self.package
        setattr(self, name, nest)


#     data[domain][arch][series] = pkg_data
#  if tag_data.comp:
#            data[tag_datadomain][arch][series][comp_tag] = pkg_data
#         self.generate_pkg("pkg", self.raw_data.model)
#  self.generate_pkg("tokenizer_pkg", self.raw_data.tokenizer)
# framework: dict[str,FrameworkBundle]
