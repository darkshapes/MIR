# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DPrepareData:
    """Represents a structured entry of the name of the class and its associated attributes."""

    model: Callable
    model_params: dict[str, list[str]] = field(init=True, default_factory=lambda: {"": [""]})

    model_name: str = field(init=False)
    library: str = field(init=False)
    import_path: str = field(init=False)

    def __post_init__(self):
        """Initializes the DPrepareData instance by setting derived attributes."""
        self.model_name: str = self.model.__name__
        self.import_path: str = self.model.__module__.rsplit(".", 1)[0]
        self.library: str = self.import_path.split(".")[0]
