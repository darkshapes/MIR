#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Register model types"""

# pylint: disable=line-too-long, import-outside-toplevel, protected-access, unsubscriptable-object

from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional
from pydantic import BaseModel, computed_field

from nnll.monitor.file import nfo, dbug, dbuq, debug_monitor
from mir.constants import CUETYPE_CONFIG, VALID_CONVERSIONS, VALID_TASKS, CueType, PkgType
from mir.mir_maid import MIRDatabase


class RegistryEntry(BaseModel):
    """Validate Hub / Ollama / LMStudio model input"""

    cuetype: CueType
    model: str
    size: int
    tags: list[str]
    timestamp: int
    api_kwargs: Optional[dict] = None
    mir: Optional[List[str]] = None
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
    def from_model_data(cls) -> list[tuple[str]]:  # lib_type: CueType) model_data: tuple[frozenset[str]]
        """Create RegistryEntry instances based on source\n
        Extract common model information and stack by newest model first for each conversion type.\n
        :param lib_type: Origin of this data (eg: HuggingFace, Ollama, CivitAI, ModelScope)
        :return: A list of RegistryEntry objects containing model metadata relevant to execution\n

        ========================================================\n
        ### GIVEN
        For any supported Cue Type:\n
        - A: Provider modules MUST be detected as installed during launch\n
        - B: Provider server MUST continue to be available, if applicable\n
        If A is **True** AND B is **True**: Library index operations will be run\n

        """
        from requests import HTTPError

        entries = []

        @CUETYPE_CONFIG.decorator
        def _read_data(data: dict = None):
            return data

        mir_db = MIRDatabase()
        api_data = _read_data()
        if CueType.check_type("HUB") or CueType.check_type("MLX_AUDIO", True):
            from huggingface_hub import scan_cache_dir, repocard, HFCacheInfo, CacheNotFound
            from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError, OfflineModeIsEnabled  # type: ignore

            try:
                model_data: HFCacheInfo = scan_cache_dir()
            except CacheNotFound:
                pass
            else:
                for repo in model_data.repos:
                    meta = {}
                    tags = []
                    package_name = None
                    mir_entry = None
                    tokenizer = None
                    mir_entry = mir_db.find_path("repo", repo.repo_id.lower())
                    try:
                        meta = repocard.RepoCard.load(repo.repo_id).data
                    except (LocalEntryNotFoundError, EntryNotFoundError, HTTPError, OfflineModeIsEnabled):
                        pass
                    if meta:
                        if hasattr(meta, "tags"):
                            tags.extend(meta.tags)
                        if hasattr(meta, "pipeline_tag"):
                            tags.append(meta.pipeline_tag)
                        test_package: str = meta.get("library_name")
                        if test_package:
                            test_package = test_package.replace("-", "_")
                            test_package.upper()
                            if hasattr(PkgType, test_package.upper()):
                                package_name = getattr(PkgType, test_package.upper())
                    if not package_name:
                        try:
                            series = mir_entry[0]
                            comp = mir_entry[1]
                            module_name = mir_db.database.get(series)
                            if module_name:
                                sub_module_name = module_name.get(comp)
                                if sub_module_name:
                                    pkg_num = sub_module_name.get("pkg")
                                    if pkg_num:
                                        pkg_id = next(iter(list(pkg_num.get(0, "diffusers"))))
                                        if hasattr(PkgType, pkg_id.upper()):
                                            package_name = getattr(PkgType, pkg_id.get.upper())
                        except (TypeError, KeyError, ValueError):
                            pass
                    if hasattr(repo, "revisions") and repo.revisions:
                        tokenizer_models = [info.file_path for info in next(iter(repo.revisions)).files if "tokenizer.json" in str(info.file_path)]
                    else:
                        tokenizer_models = None
                    tokenizer = None if not tokenizer_models else tokenizer_models[-1]
                    entry = cls(
                        model=repo.repo_id,
                        size=repo.size_on_disk,
                        tags=tags,
                        cuetype=CueType.HUB,
                        mir=mir_entry,
                        package=package_name,
                        api_kwargs=None,
                        timestamp=int(repo.last_modified),
                        tokenizer=tokenizer,
                    )
                    entries.append(entry)
                    nfo(entry.model)

        if CueType.check_type("OLLAMA", True):  # check that server is still up!
            from ollama import ListResponse, list as ollama_list

            config = api_data[CueType.OLLAMA.value[1]]
            model_data: ListResponse = ollama_list()  # type: ignore
            for model in model_data.models:  # pylint:disable=no-member
                entry = cls(
                    model=f"{api_data[CueType.OLLAMA.value[1]].get('prefix')}{model.model}",
                    size=model.size.real,
                    tags=[model.details.family],
                    cuetype=CueType.OLLAMA,
                    mir=[series for series, comp in mir_db.database.items() if model.details.family in str(comp)],
                    package=CueType.OLLAMA,
                    api_kwargs={**config["api_kwargs"]},
                    timestamp=int(model.modified_at.timestamp()),
                    tokenizer=(model.model),
                )
                entries.append(entry)

                nfo(entry.model)

        if CueType.check_type("CORTEX", True):
            import requests
            from datetime import datetime

            config = api_data[CueType.CORTEX.value[1]]
            response: requests.models.Request = requests.get(api_data["CORTEX"]["api_kwargs"]["api_base"], timeout=(3, 3))
            model: dict = response.json()
            for model_data in model["data"]:
                entry = cls(
                    model=f"{api_data[CueType.CORTEX.value[1]].get('prefix')}/{model_data.get('model')}",
                    size=model_data.get("size", 0),
                    tags=[str(model_data.get("modalities", "text"))],
                    cuetype=CueType.CORTEX,
                    mir=None,
                    package=None,
                    api_kwargs={**config["api_kwargs"]},
                    timestamp=int(datetime.timestamp(datetime.now())),  # no api for time data w openai
                    # tokenizer=None,
                )
                entries.append(entry)
                nfo(entry.model)

        if CueType.check_type("LLAMAFILE", True):
            from openai import OpenAI

            model_data: OpenAI = OpenAI(base_url=api_data["LLAMAFILE"]["api_kwargs"]["api_base"], api_key="sk-no-key-required")
            config = api_data[CueType.LLAMAFILE.value[1]]
            for model in model_data.models.list().data:
                entry = cls(
                    model=f"{api_data[CueType.LLAMAFILE.value[1]].get('prefix')}/{model.id}",
                    size=0,
                    tags=["text"],
                    cuetype=CueType.LLAMAFILE,
                    mir=None,
                    package=None,
                    api_kwargs={**config["api_kwargs"]},
                    timestamp=int(model.created),  # no api for time data w openai
                    # tokenizer=None,
                )
                entries.append(entry)
                nfo(entry.model)

        if CueType.check_type("VLLM", True):  # placeholder
            # import vllm
            config = api_data[CueType.VLLM.value[1]]
            model_data = OpenAI(base_url=api_data["VLLM"]["api_kwargs"]["api_base"], api_key=api_data["VLLM"]["api_kwargs"]["api_key"])
            for model in model_data.models.list().data:
                entry = cls(
                    model=f"{api_data[CueType.VLLM.value[1]].get('prefix')}{model['data'].get('id')}f",
                    size=0,
                    tags=["text"],
                    cuetype=CueType.VLLM,
                    mir=None,
                    package=None,
                    api_kwargs={**config["api_kwargs"]},
                    timestamp=int(model.created),  # no api for time data w openai
                    # tokenizer=None,
                )
                entries.append(entry)
                nfo(entry.model)

        if CueType.check_type("LMSTUDIO", True):
            from lmstudio import list_downloaded_models  # pylint: disable=import-error, # type: ignore

            config = api_data[CueType.LM_STUDIO.value[1]]
            model_data = list_downloaded_models()
            for model in model_data:  # pylint:disable=no-member
                tags = []
                if hasattr(model._data, "vision"):
                    tags.extend("vision", model._data.vision)
                if hasattr(model._data, "trained_for_tool_use"):
                    tags.append(("tool", model._data.trained_for_tool_use))
                entry = cls(
                    model=f"{api_data[CueType.LM_STUDIO.value[1]].get('prefix')}{model.model_key}",
                    size=model._data.size_bytes,
                    tags=tags,
                    cuetype=CueType.LM_STUDIO,
                    mir=None,
                    package=None,
                    api_kwargs={**config["api_kwargs"]},
                    timestamp=int(model.modified_at.timestamp()),
                    # tokenizer=None,
                )
                entries.append(entry)
                nfo(entry.model)

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

    # tokenizer = None if not tokenizer_models else tokenizer_models[-1]
    #     if repo.revisions:
    #         tokenizer_models = [info.file_path for info in next(iter(repo.revisions)).files if "tokenizer.json" in str(info.file_path)]
    #     else:
    #         tokenizer_models = None
    #     tokenizer = None if not tokenizer_models else tokenizer_models[-1]

    # if hasattr(repo, "revisions") and repo.revisions:
    #     tokenizer_models = [info.file_path for info in next(iter(repo.revisions)).files if "tokenizer.json" in str(info.file_path)]
    # else:
    #     tokenizer_models = None
    # tokenizer = None if not tokenizer_models else tokenizer_models[-1]
