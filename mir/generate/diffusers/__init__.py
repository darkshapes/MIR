# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable
from dataclasses import dataclass, field
from diffusers.pipelines import _import_structure as IMPORT_STRUCTURE
from diffusers.pipelines.auto_pipeline import SUPPORTED_TASKS_MAPPINGS, _get_task_class as GET_TASK_CLASS
