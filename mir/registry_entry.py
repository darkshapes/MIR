#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Register model types"""

# pylint: disable=line-too-long, import-outside-toplevel, protected-access, unsubscriptable-object

from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Union
from pydantic import BaseModel, computed_field
from mir.constants import VALID_CONVERSIONS, VALID_TASKS, CueType, PkgType
from nnll.monitor.file import dbug, dbuq


class RegistryEntry(BaseModel):
    """Validate Hub / Ollama / LMStudio model input"""

    cuetype: CueType
    model: str
    size: int
    tags: List[str]
    timestamp: int
    api_kwargs: Optional[dict] = None
    mir: Optional[List[str]] = None
    package: Optional[Union[PkgType, CueType]] = None
    tokenizer: Optional[Path] = None

    @computed_field
    @property
    def available_tasks(self) -> List[Tuple]:
        """Filter tag tasks into edge coordinates for graphing"""
        # This is a best effort at parsing tags; it is not perfect, and there is room for improvement
        # particularly: Tokenizers, being locatable here, should be assigned to their model entry
        # the logistics of how this occurs have been difficult to implement
        # additionally, tag recognition of tasks needs cleaner, which requires practical testing to solve
        import re

        default_task = None
        processed_tasks = []
        # nfo(self.model)
        if self.cuetype in [x for x in list(CueType) if x != CueType.HUB]:  # Literal list of CueType, must use list()
            default_task = ("text", "text")  # usually these are txt gen libraries
        elif self.cuetype == CueType.HUB:
            # print(self.cuetype)  # pair tags from the hub such 'x-to-y' such as 'text-to-text' etc
            pattern = re.compile(r"(\w+)-to-(\w+)")
            for tag in self.tags:
                match = pattern.search(tag)
                if match and all(group in VALID_CONVERSIONS for group in match.groups()) and (match.group(1), match.group(2)) not in processed_tasks:
                    processed_tasks.append((match.group(1), match.group(2)))
        for tag in self.tags:  # when pair-tagged elements are not available, potential to duplicate HUB tags here
            for (graph_src, graph_dest), tags in VALID_TASKS[self.cuetype].items():
                if tag in tags and (graph_src, graph_dest) not in processed_tasks:
                    processed_tasks.append((graph_src, graph_dest))
        if default_task and default_task not in processed_tasks:
            processed_tasks.append(default_task)
        return processed_tasks

    @classmethod
    def create_entry(
        cls,
        model: str,
        size: int,
        tags: List[str],
        cuetype: CueType,
        mir: Optional[List[str]] = None,
        package: Optional[Union[PkgType, CueType]] = None,
        api_kwargs=None,
        timestamp: Optional[int] = None,
        tokenizer=Optional[str],
    ):
        """API specific data to call models\n
        :param model:Cache location for model
        :param size: File size (usually in bytes)
        :param tags: List of available machine tasks for model
        :param cuetype: Provider to trigger loading
        :param mir: MIR information, defaults to None
        :param package: Package name and availability, defaults to None
        :param api_kwargs: Localhost server defaults, defaults to None
        :param timestamp: Download time of model, defaults to None
        :param tokenizer: Tokenizer configuration location, defaults to None
        :return: An instance of RegistryEntry with the provided values
        """
        from datetime import datetime

        entry = cls(
            model=model,
            size=size,
            tags=tags,
            cuetype=cuetype,
            mir=mir,
            package=package,
            api_kwargs=api_kwargs,
            timestamp=timestamp or int(datetime.now().timestamp()),  # Default to current time if not provided
            tokenizer=tokenizer,
        )
        dbuq(entry)
        return entry
