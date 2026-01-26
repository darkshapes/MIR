# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from diffusers.pipelines import _import_structure as IMPORT_STRUCTURE
from diffusers.pipelines.auto_pipeline import SUPPORTED_TASKS_MAPPINGS
from diffusers.pipelines.auto_pipeline import _get_task_class as GET_TASK_CLASS

from typing import get_type_hints


class GatherLoop:
    def __init__(self) -> None:
        """Loops through diffusers packages to harvest class data."""
        from mir.maid import MIRDatabase

        self.db = MIRDatabase()
        from mir.data import EXCLUSIONS
        from mir.build_entry import BuildEntry

        build_entries = []
        subclasses = self.extract_subclass_data("diffusers", "DiffusionPipeline")
        for module_path, pipeline in subclasses.items():
            if module_path.rsplit(".", 1)[-1] not in EXCLUSIONS["exclusion_list"]:
                build_entries.extend([BuildEntry(model_type=model_type, model=model) for model_type, model in get_type_hints(pipeline.__init__).items()])
            build_entries.append(BuildEntry(model_type="pipeline", model=pipeline))
        self.model_db = {x.attributes.model_name: x.attributes.model.layers for x in build_entries for x in build_entries if hasattr(x.attributes, "layers")}
        # TODO: for data in prepared_data:

    def extract_subclass_data(self, package_name: str, base_class_name: str):
        """Extracts subclasses from a package that inherit from a specified base class.\n
        :param package_name: Name of the package to search
        :param base_class_name: Name of the base class to inherit from
        :return: Dictionary mapping fully qualified class names to class objects"""

        from pkgutil import walk_packages
        from inspect import getmro
        from importlib import import_module

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
