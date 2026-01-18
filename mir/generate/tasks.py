# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from typing import Any, Callable, List, get_type_hints
from mir.maid import MIRDatabase
from mir.generate.diffusers.index import show_diffusers_tasks
from mir.generate.diffusers.schedulers import tag_scheduler
from mir import DBUQ
from mir.tag import MIRTag

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

    async def detect_pipes(self, mir_tag: MIRTag, model: Callable, type_params: dict) -> dict:
        """Detects and traces Pipes MIR data\n
        :param mir_db:: An instance of MIRDatabase containing the database of information.
        :type mir_db: MIRDatabase
        :param field_name:  The name of the field in compatibility data to process for task detection, defaults to "pkg".
        :type field_name: str, optional
        :return:A dictionary mapping series names to their respective compatibility and traced tasks.
        :rtype: dict"""

        data_tuple = []
        tasks = show_diffusers_tasks(code_name= class_name=model.__name__)
        detected_pipe = await self.hyperlink_to_mir(type_params, mir_tag.series)
        if hasattr(mir_tag, "comp") and mir_tag.comp:
            data_tuple.append((*mir_tag.series, {mir_tag.comp: detected_pipe}))
        else:
            data_tuple.append(({mir_tag.series: detected_pipe}))

        return data_tuple

    async def hyperlink_to_mir(self, pipe_args: dict, series: str):
        """Maps pipeline components to MIR tags/IDs based on class names and roles.\n
        :param pipe_args: Dictionary of pipeline roles to their corresponding classes
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
                            mir_tag, class_name = await self.tag_class(pipe_class=union_class, pipe_role=pipe_role, series=series)
                            # mir_tag = mir_db.find_tag(field="tasks", target=class_name)
                            # dbuq(f"{mir_tag} {class_name}")
                        detected_links["pipe_names"][pipe_role].append(mir_tag if mir_tag else class_name)
                else:
                    mir_tag, class_name = await self.tag_class(pipe_class=pipe_class, pipe_role=pipe_role, series=series)
                    detected_links["pipe_names"][pipe_role] = mir_tag if mir_tag else [class_name]
                    mir_tag = None
                    class_name = None
        return detected_links

    async def tag_class(self, pipe_class: Callable, pipe_role: str, series: str) -> tuple[str | None]:
        """Maps a class to MIR tags/IDs based on its name and role.\n
        :param pipe_class: Class to be mapped
        :param pipe_role: Role of the class in the pipeline
        :param series: Series identifier for the component
        :return: Tuple containing MIR tag and class name"""

        mir_tag = None
        class_name = pipe_class.__name__
        if pipe_role in ["scheduler", "image_noising_scheduler", "prior_scheduler"]:
            sub_field = pipe_class.__module__.split(".")[0]
            scheduler_series, scheduler_comp = tag_scheduler(class_name)
            mir_tag = [f"ops.scheduler.{scheduler_series}", scheduler_comp]
            DBUQ(f"scheduler {mir_tag} {class_name} {sub_field} ")
        elif pipe_role == "vae":
            sub_field = pipe_class.__module__.split(".")[0]
            mir_comp = series.rsplit(".", 1)[-1]
            DBUQ(mir_comp)
            mir_tag = "info.vae"
        return mir_tag, class_name

