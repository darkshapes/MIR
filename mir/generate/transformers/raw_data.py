# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable
from dataclasses import dataclass, field


@dataclass
class PrepareData:
    """Represents a structured entry of the name of the class and its associated attributes."""

    model: Callable
    config_params: dict[str, list[str]]
    config: Callable | None = None

    model_name: str = field(init=False)
    library: str = field(init=False)
    import_path: str = field(init=False)

    def __post_init__(self) -> None:
        """Initializes the PrepareData instance by setting derived attributes."""
        self.model_name: str = self.model.__name__
        self.import_path = self.model.__module__.rsplit(".", 1)[0]
        self.library = self.import_path.split(".")[0]
