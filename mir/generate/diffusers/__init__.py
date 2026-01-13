# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from dataclasses import dataclass
from typing import Callable
from diffusers.pipelines import _import_structure as IMPORT_STRUCTURE
from diffusers.pipelines.auto_pipeline import SUPPORTED_TASKS_MAPPINGS, _get_task_class as GET_TASK_CLASS


@dataclass
class DocStringEntry:
    """Represents a structured entry of package name, file name, and docstring."""

    package_name: str
    doc_string: str
    file_name: str
    pipe_module: Callable


class DocParseData:
    pipe_class: str
    pipe_repo: str
    staged_class: str | None = None
    staged_repo: str | None = None

    def __init__(self, pipe_class: str, pipe_repo: str, staged_class: str | None = None, staged_repo: str | None = None):
        self.pipe_class = pipe_class
        self.pipe_repo = pipe_repo
        self.staged_class = staged_class
        self.staged_repo = staged_repo
