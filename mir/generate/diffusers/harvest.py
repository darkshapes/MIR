# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from importlib import import_module
from pkgutil import walk_packages
from inspect import getmro

from mir.framework import MIRNesting
from mir.generate.diffusers.raw_data import DPrepareData
from mir.tag import MIRTag


class HarvestClasses:
    def __init__(self) -> None:
        self.parsed_docs = []
        pass

    def find_diffusers_docstrings(self) -> None:
        """Pull down docstrings from 🤗Diffusers pipelines, minimizing internet requests\n
        :return: Docstrings for common diffusers models"""

        self.extract_model_data()

    def extract_model_data(self):
        from mir.generate.diffusers import EXCLUSIONS
        from mir.generate.tasks import TaskAnalyzer

        subclasses = self.subclasses_of("diffusers", "DiffusionPipeline")
        for path, class_obj in subclasses.items():
            if path.rsplit(".", 1)[-1] in EXCLUSIONS["exclusion_list"].get("model_path", "."):
                continue
            base_path = path.rsplit(".", 1)[0]
            model_path = import_module(base_path)
            if doc_string := getattr(model_path, "EXAMPLE_DOC_STRING", None):
                prepared_data = DPrepareData(doc_string=doc_string, model=class_obj, model_path=base_path)
                mir_tag = MIRTag(prepared_data)
                task_analysis = TaskAnalyzer()
                mir_nest = MIRNesting(mir_tag, prepared_data)

    def subclasses_of(self, package_name: str, base_class_name: str):
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

        results = {}
        root_pkg = import_module(package_name)
        for finder, mod_name, is_pkg in walk_packages(root_pkg.__path__, root_pkg.__name__ + "."):
            try:
                module = import_module(mod_name)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue

            # Iterate over all *already* imported members in the module
            for name, obj in module.__dict__.items():
                if not isinstance(obj, type):
                    continue
                # Ensure the class is defined in this module, not imported
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

    # def extract_model_data(self,pipe_name, file_name: str) -> dict | None:
    #     migrated_pipes = MIGRATIONS["migrated_pipes"]
    #     pkg_path = f"diffusers.pipelines.{pipe_name}.{file_name}"
    #     pipe_file: Callable = import_object_named(file_name, pkg_path) or import_module(pkg_path)
    #     if pipe_file and (doc_string := getattr(pipe_file, "EXAMPLE_DOC_STRING", None)): #where pipe class and repo are
    #         docstrings= DocStringEntry(package_name=pipe_name, file_name=file_name, pipe_module=pipe_file, doc_string=doc_string)
    #         DocStringParser(doc_string=docstrings.doc_string)
    #         self.parsed_docs.pipe_repo = migrated_pipes.get(self.parsed_docs.pipe_class, self.parsed_docs.pipe_repo)
    #         model = import_object_named(parsed_data.pipe_class, docstrings.pipe_module.__name__)
    #         model_data = show_init_fields_for(model,"diffusers")
    #         return {"model_params": model_data}


#   for pipe_name in IMPORT_STRUCTURE.keys():
#             if pipe_name not in exclusion_list and (import_name := getattr(diffusers_pipelines, str(pipe_name))):
#                 file_specific = uncommon_naming.get(pipe_name, pipe_name)
#                 file_names:list[str] = [getattr(import_name, "_import_structure", {})] or [f"pipeline_{file_specific}"]
#                 for file_name in file_names:
#                     if not file_name in exclusion_list or not (model_data := self.extract_model_data(pipe_name, file_name)):
#                         continue
#                     if not (prepared_data := PrepareData( **model_data)):
#                         continue
# else:
# continue
