#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Register model types"""

# pylint: disable=line-too-long, import-outside-toplevel, protected-access, unsubscriptable-object

from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional
from pydantic import BaseModel, computed_field

from nnll.monitor.file import dbug, debug_monitor
from mir.constants import LIBTYPE_CONFIG, VALID_CONVERSIONS, VALID_TASKS, LibType, PkgType
from mir.mir_maid import MIRDatabase


class RegistryEntry(BaseModel):
    """Validate Hub / Ollama / LMStudio model input"""

    model: str
    size: int
    tags: list[str]
    library: LibType
    timestamp: int
    mir: Optional[list[str]] = None
    api_kwargs: Optional[dict] = None
    package: Optional[Enum] = None
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
        library_tasks = {}
        processed_tasks = []
        library_tasks = VALID_TASKS[self.library]
        if self.library in [x for x in list(LibType) if x != LibType.HUB]:
            default_task = ("text", "text")  # usually these are txt gen libraries
        elif self.library == LibType.HUB:
            # print(self.library)  # pair tags from the hub such 'x-to-y' such as 'text-to-text' etc
            pattern = re.compile(r"(\w+)-to-(\w+)")
            for tag in self.tags:
                match = pattern.search(tag)
                if match and all(group in VALID_CONVERSIONS for group in match.groups()) and (match.group(1), match.group(2)) not in processed_tasks:
                    processed_tasks.append((match.group(1), match.group(2)))
        for tag in self.tags:  # when pair-tagged elements are not available, potential to duplicate HUB tags here
            for (graph_src, graph_dest), tags in library_tasks.items():
                if tag in tags and (graph_src, graph_dest) not in processed_tasks:
                    processed_tasks.append((graph_src, graph_dest))
        if default_task and default_task not in processed_tasks:
            processed_tasks.append(default_task)
        return processed_tasks

    @classmethod
    def from_model_data(cls) -> list[tuple[str]]:  # lib_type: LibType) model_data: tuple[frozenset[str]]
        """Create RegistryEntry instances based on source\n
        Extract common model information and stack by newest model first for each conversion type.\n
        :param lib_type: Origin of this data (eg: HuggingFace, Ollama, CivitAI, ModelScope)
        :return: A list of RegistryEntry objects containing model metadata relevant to execution\n

        ========================================================\n
        ### GIVEN
        For any supported Library Type:\n
        - A: Library modules MUST be detected as installed during launch\n
        - B: Library server MUST continue to be available\n
        If A is **True** AND B is **True**: Library index operations will be run\n

        """
        entries = []

        @LIBTYPE_CONFIG.decorator
        def _read_data(data: dict = None):
            return data

        mir_db = MIRDatabase()
        api_data = _read_data()
        if LibType.check_type("HUB"):
            from huggingface_hub import scan_cache_dir, repocard, HFCacheInfo, CacheNotFound  # type: ignore

            try:
                model_data: HFCacheInfo = scan_cache_dir()
                for repo in model_data.repos:
                    try:
                        meta = repocard.RepoCard.load(repo.repo_id).data
                    except ValueError as error_log:
                        dbug(error_log)
                    else:
                        package_name = meta.get("library_name")
                        if package_name:
                            package_name = package_name.replace("-", "_").upper()
                            if hasattr(PkgType, package_name):
                                package_name = getattr(PkgType, package_name.upper())
                        tags = []
                        tokenizer_models = [info.file_path for info in next(iter(repo.revisions)).files if "tokenizer.json" in str(info.file_path)]
                        tokenizer = None if not tokenizer_models else tokenizer_models[-1]
                        if hasattr(meta, "tags"):
                            tags.extend(meta.tags)
                        if hasattr(meta, "pipeline_tag"):
                            tags.append(meta.pipeline_tag)
                        entry = cls(
                            model=repo.repo_id,
                            size=repo.size_on_disk,
                            tags=tags,
                            library=LibType.HUB,
                            mir=mir_db.find_path("repo", repo.repo_id.lower()),
                            package=package_name,
                            api_kwargs=None,
                            timestamp=int(repo.last_modified),
                            tokenizer=tokenizer,
                        )  # pylint: disable=undefined-loop-variable
                        entries.append(entry)
            except CacheNotFound as error_log:
                dbug(error_log)

        if LibType.check_type("OLLAMA"):  # check that server is still up!
            from ollama import ListResponse, list as ollama_list

            config = api_data[LibType.OLLAMA.value[1]]
            model_data: ListResponse = ollama_list()  # type: ignore
            for model in model_data.models:  # pylint:disable=no-member
                entry = cls(
                    model=f"{api_data[LibType.OLLAMA.value[1]].get('prefix')}{model.model}",
                    size=model.size.real,
                    tags=[model.details.family],
                    library=LibType.OLLAMA,
                    mir=[series for series, comp in mir_db.database.items() if model.details.family in str(comp)],
                    package=LibType.OLLAMA,
                    api_kwargs={**config["api_kwargs"]},
                    timestamp=int(model.modified_at.timestamp()),
                    tokenizer=(model.model),
                )
                entries.append(entry)

        if LibType.check_type("CORTEX"):
            import requests
            from datetime import datetime

            config = api_data[LibType.CORTEX.value[1]]
            response: requests.models.Request = requests.get(api_data["CORTEX"]["api_kwargs"]["api_base"], timeout=(3, 3))
            model: dict = response.json()
            for model_data in model["data"]:
                entry = cls(
                    model=f"{api_data[LibType.CORTEX.value[1]].get('prefix')}/{model_data.get('model')}",
                    size=model_data.get("size", 0),
                    tags=[str(model_data.get("modalities", "text"))],
                    library=LibType.CORTEX,
                    mir=None,
                    package=LibType.CORTEX,
                    api_kwargs={**config["api_kwargs"]},
                    timestamp=int(datetime.timestamp(datetime.now())),  # no api for time data in cortex
                )
                entries.append(entry)

        if LibType.check_type("LLAMAFILE"):
            from openai import OpenAI

            model_data: OpenAI = OpenAI(base_url=api_data["LLAMAFILE"]["api_kwargs"]["api_base"], api_key="sk-no-key-required")
            config = api_data[LibType.LLAMAFILE.value[1]]
            for model in model_data.models.list().data:
                entry = cls(
                    model=f"{api_data[LibType.LLAMAFILE.value[1]].get('prefix')}/{model.id}",
                    size=0,
                    tags=["text"],
                    library=LibType.LLAMAFILE,
                    mir=None,
                    package=LibType.LLAMAFILE,
                    api_kwargs={**config["api_kwargs"]},
                    timestamp=int(model.created),  # no api for time data in cortex
                )
                entries.append(entry)

        if LibType.check_type("VLLM"):  # placeholder
            # import vllm
            config = api_data[LibType.VLLM.value[1]]
            model_data = OpenAI(base_url=api_data["VLLM"]["api_kwargs"]["api_base"], api_key=api_data["VLLM"]["api_kwargs"]["api_key"])
            for model in model_data.models.list().data:
                entry = cls(
                    model=f"{api_data[LibType.VLLM.value[1]].get('prefix')}{model['data'].get('id')}f",
                    size=0,
                    tags=["text"],
                    library=LibType.VLLM,
                    mir=None,
                    package=LibType.VLLM,
                    api_kwargs={**config["api_kwargs"]},
                    timestamp=int(model.created),  # no api for time data in cortex
                )
                entries.append(entry)

        if LibType.check_type("LMSTUDIO"):
            from lmstudio import list_downloaded_models  # pylint: disable=import-error, # type: ignore

            config = api_data[LibType.LM_STUDIO.value[1]]
            model_data = list_downloaded_models()
            for model in model_data:  # pylint:disable=no-member
                tags = []
                if hasattr(model._data, "vision"):
                    tags.extend("vision", model._data.vision)
                if hasattr(model._data, "trained_for_tool_use"):
                    tags.append(("tool", model._data.trained_for_tool_use))
                entry = cls(
                    model=f"{api_data[LibType.LM_STUDIO.value[1]].get('prefix')}{model.model_key}",
                    size=model._data.size_bytes,
                    tags=tags,
                    library=LibType.LM_STUDIO,
                    mir=None,
                    package=LibType.LM_STUDIO,
                    api_kwargs={**config["api_kwargs"]},
                    timestamp=int(model.modified_at.timestamp()),
                )
                entries.append(entry)
        return sorted(entries, key=lambda x: x.timestamp, reverse=True)


@debug_monitor
def from_cache() -> list[str, RegistryEntry]:
    """
    Retrieve models from ollama server, local huggingface hub cache, local lmstudio cache & vllm.
    我們不應該繼續為LMStudio編碼。 歡迎貢獻者來改進它。 LMStudio is not OSS, but contributions are welcome.
    """
    models = None
    models = RegistryEntry.from_model_data()
    dbug(f"REG_ENTRIES {models}")
    return models
