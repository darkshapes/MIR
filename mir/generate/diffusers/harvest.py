# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from importlib import import_module
from inspect import getmro
from typing import Any, Callable, get_type_hints

from mir.generate.diffusers.raw_data import DPrepareData
from mir.package import MIRNesting, MIRPackage
from mir.tag import MIRTag


class HarvestClasses:
    def __init__(self) -> None:
        """Initializes the HarvestClasses instance with an empty list to store raw class data."""
        from mir.maid import MIRDatabase

        self.db = MIRDatabase()
        self.raw_data = []
        self.find_diffusers_docstrings()

    def find_diffusers_docstrings(self) -> None:
        """Pull down docstrings from 🤗Diffusers pipelines, minimizing internet requests\n
        :return: Docstrings for common diffusers models"""

        # from mir.generate.tasks import TaskAnalyzer

        subclasses = self.extract_subclass_data("diffusers", "DiffusionPipeline")
        for module_path, model in subclasses.items():
            if not (base_data := self.extract_base_data(module_path)):
                continue
            if not (model_data := self.extract_model_class_data(model)):
                continue
            if not (prepared_data := DPrepareData(**base_data, **model_data)):
                continue
            mir_tag = MIRTag(prepared_data)
            # task_analysis = TaskAnalyzer(prepared_data=prepared_data, mir_tag=mir_tag)
            mir_nest = MIRNesting(mir_tag, prepared_data)
            packages = {"model": MIRPackage(data=prepared_data.model)}
            for component_name, component_model in prepared_data.model_params.items():
                if hasattr(prepared_data, component_name):
                    packages.setdefault(component_name, MIRPackage(data=component_model))
            packages.setdefault("framework", MIRPackage(data=mir_nest.framework_data))
            # print(packages)
            mir_nest(packages)

            self.db.add_data(mir_nest, *mir_nest.loops)

    def extract_base_data(self, module_path: str) -> dict[str, str] | None:
        from mir.data import EXCLUSIONS

        if module_path.rsplit(".", 1)[-1] in EXCLUSIONS["exclusion_list"]:
            return None
        base_path = module_path.rsplit(".", 1)[0]
        model_path = import_module(base_path)
        if doc_string := getattr(model_path, "EXAMPLE_DOC_STRING", None):
            return {
                "doc_string": doc_string,
                "model_path": base_path,
            }
        return None

    def extract_model_class_data(self, model: Callable) -> dict[str, str | Callable | dict[str, Any]] | None:
        model_name: str = model.__name__
        library: str = model.__module__.split(".", 1)[0]
        model_params: dict[str, Any] = get_type_hints(model.__init__)
        for module in model_params.values():
            module_name = module.__module__
            library_path = f"{library}.models."
            if library_path in module_name:
                module_name = module_name.replace(library_path, "").split(".")[0]
                return {
                    "model": model,
                    "model_name": model_name,
                    "model_params": model_params,
                    "library": library,
                }
        return None

    def extract_subclass_data(self, package_name: str, base_class_name: str):
        """
        Return a dict mapping `<module_name>.<class_name>` → class object
        for every class in `package_name` that subclasses a class named
        `base_class_name`.

        The implementation is intentionally defensive: it avoids
        triggering `__getattr__` on lazy‑loaded submodules that might
        raise a `RuntimeError`.  Instead of `inspect.getmembers`, it
        iterates over the module's `__dict__` which contains only
        attributes that have already been imported.
        """
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
