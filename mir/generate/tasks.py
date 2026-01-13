# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from typing import Any, Callable, List, get_type_hints
from mir.generate.from_module import get_internal_name_for, import_object_named
from mir.generate.transformers.index import show_transformers_tasks
from mir.maid import MIRDatabase
from mir.generate.diffusers.index import show_diffusers_tasks
from mir.generate.diffusers.schedulers import tag_scheduler
from mir import DBUQ

flatten_map: List[Any] = lambda nested, unpack: [element for iterative in getattr(nested, unpack)() for element in iterative]
flatten_map.__annotations__ = {"nested": List[str], "unpack": str}


class TaskAnalyzer:
    def __init__(self) -> None:
        self.skip_series = [
            "info.lora",
            "info.vae",
            "ops.precision",
            "ops.scheduler",
            "info.encoder.tokenizer",
            "info.controlnet",
        ]
        self.skip_classes = [".gligen", "imagenet64"]
        self.skip_auto = ["AutoTokenizer", "AutoModel", "AutoencoderTiny", "AutoencoderKL", "AutoPipelineForImage2Image"]
        self.skip_types = ["int", "bool", "float", "Optional", "NoneType", "List", "UNet2DConditionModel"]
        self.mflux_tasks = ["Image", "Redux", "Kontext", "Depth", "Fill", "ConceptAttention", "ControlNet", "CavTon", "IC-Edit"]

    async def detect_tasks(self, mir_db: MIRDatabase, field_name: str = "pkg") -> dict:
        """Detects and traces tasks MIR data\n
        :param mir_db:: An instance of MIRDatabase containing the database of information.
        :type mir_db: MIRDatabase
        :param field_name:  The name of the field in compatibility data to process for task detection, defaults to "pkg".
        :type field_name: str, optional
        :return: A dictionary mapping series names to their respective compatibility and traced tasks.
        :rtype: dict"""

        data_tuple = []
        for series, compatibility_data in mir_db.database.items():
            if (
                series.startswith("info.")  # formatting comment
                and not any(tag for tag in self.skip_series if series.startswith(tag))
                and not any(tag for tag in self.skip_classes if tag in series)
            ):
                for compatibility, field_data in compatibility_data.items():
                    if field_data and field_data.get(field_name, {}).get("0"):
                        tasks_for_class = {"tasks": []}
                        for _, pkg_tree in field_data[field_name].items():
                            detected_tasks = await self.trace_tasks(pkg_tree=pkg_tree)
                            if detected_tasks:
                                for task in detected_tasks:
                                    if task not in tasks_for_class["tasks"]:
                                        tasks_for_class["tasks"].append(task)
                                data_tuple.append((*series.rsplit(".", 1), {compatibility: tasks_for_class}))

        return data_tuple

    async def detect_pipes(self, mir_db: MIRDatabase, field_name: str = "pkg") -> dict:
        """Detects and traces Pipes MIR data\n
        :param mir_db:: An instance of MIRDatabase containing the database of information.
        :type mir_db: MIRDatabase
        :param field_name:  The name of the field in compatibility data to process for task detection, defaults to "pkg".
        :type field_name: str, optional
        :return:A dictionary mapping series names to their respective compatibility and traced tasks.
        :rtype: dict"""

        data_tuple = []
        for series, compatibility_data in mir_db.database.items():
            if (
                series.startswith("info.")  # formatting comment
                and not any(series.startswith(tag) for tag in self.skip_series)
                and not any(tag for tag in self.skip_classes if tag in series)
            ):
                for compatibility, field_data in compatibility_data.items():
                    if field_data and field_data.get(field_name, {}).get("0"):
                        for _, pkg_tree in field_data[field_name].items():
                            if pkg_tree and next(iter(pkg_tree)) == "diffusers":
                                module_name = pkg_tree[next(iter(pkg_tree))]
                                DBUQ(f"{module_name} pipe originator")
                                class_obj = import_object_named(module_name, "diffusers")
                                pipe_args = get_type_hints(class_obj.__init__)
                                detected_pipe = await self.hyperlink_to_mir(pipe_args, series, mir_db)
                                data_tuple.append((*series.rsplit(".", 1), {compatibility: detected_pipe}))

        return data_tuple

    async def hyperlink_to_mir(self, pipe_args: dict, series: str, mir_db: MIRDatabase):
        """Maps pipeline components to MIR tags/IDs based on class names and roles.\n
        :param pipe_args: Dictionary of pipeline roles to their corresponding classes
        :param mir_db: MIRDatabase instance for querying tags/IDs
        :return: Dictionary mapping pipeline roles to associated MIR tags/IDs"""

        mir_tag: None | list[str] = None
        detected_links: dict[str, dict] = {"pipe_names": dict()}
        for pipe_role, pipe_class in pipe_args.items():
            if pipe_role in ["tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4", "prior_tokenizer"]:
                detected_links["pipe_names"].setdefault(pipe_role, ["info.encoder.tokenizer", series.rsplit(".", 1)[-1]])
                continue
            if not any(segment for segment in self.skip_types if pipe_class.__name__ == segment):
                mir_tag = None
                detected_links["pipe_names"][pipe_role] = []
                DBUQ(f"pipe_class.__name__ {pipe_class.__name__} {pipe_class}")
                if pipe_class.__name__ in ["Union"]:
                    for union_class in pipe_class.__args__:
                        mir_tag = None
                        class_name = union_class.__name__
                        if not any(segment for segment in self.skip_types if class_name == segment):
                            mir_tag, class_name = await self.tag_class(pipe_class=union_class, pipe_role=pipe_role, series=series, mir_db=mir_db)
                            # mir_tag = mir_db.find_tag(field="tasks", target=class_name)
                            # dbuq(f"{mir_tag} {class_name}")
                        detected_links["pipe_names"][pipe_role].append(mir_tag if mir_tag else class_name)
                else:
                    mir_tag, class_name = await self.tag_class(pipe_class=pipe_class, pipe_role=pipe_role, series=series, mir_db=mir_db)
                    detected_links["pipe_names"][pipe_role] = mir_tag if mir_tag else [class_name]
                    mir_tag = None
                    class_name = None
        return detected_links

    async def tag_class(self, pipe_class: Callable, pipe_role: str, series: str, mir_db: MIRDatabase) -> tuple[str | None]:
        """Maps a class to MIR tags/IDs based on its name and role.\n
        :param pipe_class: Class to be mapped
        :param pipe_role: Role of the class in the pipeline
        :param series: Series identifier for the component
        :param mir_db: MIRDatabase instance for querying tags/IDs
        :return: Tuple containing MIR tag and class name"""

        mir_tag = None
        class_name = pipe_class.__name__
        if pipe_role in ["scheduler", "image_noising_scheduler", "prior_scheduler"]:
            sub_field = pipe_class.__module__.split(".")[0]
            scheduler_series, scheduler_comp = tag_scheduler(class_name)
            mir_tag = [f"ops.scheduler.{scheduler_series}", scheduler_comp]
            if not mir_db.database.get(mir_tag[0], {}).get(mir_tag[1]):
                mir_tag = mir_db.find_tag(field="pkg", target=class_name, sub_field=sub_field, domain="ops.scheduler")
            DBUQ(f"scheduler {mir_tag} {class_name} {sub_field} ")
        elif pipe_role == "vae":
            sub_field = pipe_class.__module__.split(".")[0]
            mir_comp = series.rsplit(".", 1)[-1]
            DBUQ(mir_comp)
            mir_tag = [mir_id for mir_id, comp_data in mir_db.database.items() if "info.vae" in mir_id and next(iter(comp_data)) == mir_comp]
            if mir_tag:
                mir_tag.append(mir_comp)  # keep mir tag as single list
            elif class_name != "AutoencoderKL":
                DBUQ(pipe_class)
                mir_tag = mir_db.find_tag(field="pkg", target=class_name, sub_field=sub_field, domain="info.vae")
            DBUQ(f"vae {mir_tag} {class_name} {sub_field} ")
        else:
            mir_tag = mir_db.find_tag(field="tasks", target=class_name)
        return mir_tag, class_name

    async def trace_tasks(self, pkg_tree: dict[str, str | int | list[str | int]]) -> List[str]:
        """Trace tasks for a given MIR entry.\n
        :param entry: The object containing the model information.
        :return: A sorted list of tasks applicable to the model."""

        preformatted_task_data = None
        filtered_tasks = None
        snip_words: set[str] = {"load_tf_weights_in"}
        package_name = next(iter(pkg_tree))
        DBUQ(pkg_tree)
        class_name = pkg_tree[package_name]
        DBUQ(f"{package_name}, {class_name}")
        if class_name not in self.skip_auto:
            if isinstance(class_name, dict):
                class_name = next(iter(list(class_name)))
            if package_name == "transformers":
                preformatted_task_data = show_transformers_tasks(class_name=class_name)
            elif package_name == "diffusers":
                code_name = get_internal_name_for(class_name, package_name)
                preformatted_task_data = show_diffusers_tasks(code_name=code_name, class_name=class_name)
                preformatted_task_data.sort()
            elif package_name == "mflux":
                preformatted_task_data = self.mflux_tasks
            if preformatted_task_data:
                filtered_tasks = [task for task in preformatted_task_data for snip in snip_words if snip not in task]
                return filtered_tasks  # package_name, class_name
