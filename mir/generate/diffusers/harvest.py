# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from importlib import import_module
from inspect import getmro
from typing import get_type_hints

from mir.generate.diffusers.raw_data import DPrepareData


class HarvestLoop:
    def __init__(self) -> None:
        """Initializes the HarvestClasses instance with an empty list to store raw class data."""
        from mir.generate.transformers.harvest import HarvestLoop

        from mir.maid import MIRDatabase

        self.db = MIRDatabase()
        self.harvest_tf = HarvestLoop()

    def __call__(self) -> None:
        from mir.data import EXCLUSIONS

        prepared_data = {}
        library = "diffusers"
        subclasses = self.extract_subclass_data(library, "DiffusionPipeline")  # diffusers.pipelines.
        for module_path, pipeline in subclasses.items():
            if module_path.rsplit(".", 1)[-1] not in EXCLUSIONS["exclusion_list"]:
                loop_parameters = get_type_hints(pipeline.__init__)
                loop_parameters.setdefault("pipeline", pipeline)
                for name, self.model in loop_parameters.items():
                    if prepare_data := self.prepare_class_data():
                        prepared_data.setdefault(name, prepare_data)
        for data in prepared_data:
            pass

    def prepare_class_data(self):
        prepared_data = DPrepareData(model=self.model)
        return prepared_data

    def extract_subclass_data(self, package_name: str, base_class_name: str):
        """Return a dict mapping `<module_name>.<class_name>` → class object
        for every class in `package_name` that subclasses a class named
        `base_class_name`."""

        from pkgutil import walk_packages

        results = {}
        root_pkg = import_module(package_name)
        for finder, mod_name, is_pkg in walk_packages(root_pkg.__path__, root_pkg.__name__ + "."):
            try:
                module = import_module(mod_name)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue

            for name, obj in module.__dict__.items():
                if not isinstance(obj, type):
                    continue
                if obj.__module__ != mod_name:
                    continue
                try:
                    bases = getmro(obj)[1:]  # skip the class itself
                except ValueError:
                    continue
                for base in bases:
                    if base.__name__ == base_class_name:
                        fqcn = f"{mod_name}.{name}"
                        results[fqcn] = obj
                        break

        return results
