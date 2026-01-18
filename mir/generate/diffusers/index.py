# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from importlib import import_module
from typing import Any, Generator

from mir import DBUQ, NFO
from mir.data import EXCLUSIONS
from mir.generate.diffusers import GET_TASK_CLASS, IMPORT_STRUCTURE, SUPPORTED_TASKS_MAPPINGS
from mir.generate.from_module import import_object_named, show_init_fields_for, to_domain_tag
from mir.generate.indexers import migrations


def retrieve_diffusers_docstrings(
    package_name: str,
    file_names: list[str],
) -> Generator[DocStringEntry]:
    """Yield (pkg, file, EXAMPLE_DOC_STRING) from a folder or a single file.\n
    :param pkg_name: Package under ``diffusers.pipelines``.\n
    :param file_names: A list of related file names.\n
    :param use_folder: True → treat ``source`` as a folder with ``_import_structure``.\n
    :return: DocString Entry class.\n
    """

    module_location: str | None = import_module("diffusers.pipelines").__file__
    module_path = os.path.dirname(module_location)

    for file_name in file_names:
        assert isinstance(file_name, str), f"Expected path to be string, got {file_name} type {type(file_name)}"
        if file_name == "pipeline_stable_diffusion_xl_inpaint":
            continue

        pkg_path = f"diffusers.pipelines.{package_name}.{file_name}"
        DBUQ(pkg_path)

        if os.path.exists(os.path.join(module_path, package_name, f"{file_name}.py")):
            pipe_file = import_object_named(file_name, pkg_path) or import_module(pkg_path) or NFO(f"Failed to import {pkg_path}")
            if doc_string := getattr(pipe_file, "EXAMPLE_DOC_STRING", None):
                yield DocStringEntry(package_name=package_name, file_name=file_name, pipe_module=pipe_file, doc_string=doc_string)
            else:
                NFO(f"Doc string attribute missing for {package_name}/{file_name}")
        else:
            NFO(f"Path not found for {package_name}/{file_name}")

    return


def create_pipe_entry(repo_path: str, class_name: str, model_class_obj: Callable | None = None) -> tuple[str, dict[str, dict[Any, Any]]]:
    """Create a pipeline article and generate corresponding information according to the provided repo path and pipeline category\n
    :param repo_path (str): Repository path.
    :param model_class_obj (str): The model class function
    :raises TypeError: If 'repo_path' or 'class_name' are not set.
    :return: Tuple: The data structure containing mir_series and mir_comp is used for subsequent processing.
    """
    import diffusers  # pyright: ignore[reportMissingImports] # pylint:disable=redefined-outer-name

    control_net = ["Control", "Controlnet"]  #
    mir_prefix = "info"
    if hasattr(diffusers, class_name):
        model_class_obj = getattr(diffusers, class_name)
        sub_segments = show_init_fields_for(model_class_obj, "diffusers")
        decoder = "decoder" in sub_segments
        if repo_path in ["kandinsky-community/kandinsky-3"]:
            mir_prefix = "info.unet"
        if repo_path in ["openai/shap-e"]:
            mir_prefix = "info.unet"
            class_name = "ShapEPipeline"
        elif class_name == "MotionAdapter":
            mir_prefix = "info.lora"
        elif class_name == "WanPipeline":
            mir_prefix = "info.dit"
        elif class_name == "CogVideoXVideoToVideoPipeline":
            class_name = "CogVideoXPipeline"
        elif any(maybe for maybe in control_net if maybe.lower() in class_name.lower()):
            mir_prefix = "info.controlnet"
        else:
            mir_prefix = to_domain_tag(**sub_segments)
            if mir_prefix is None and class_name not in ["AutoPipelineForImage2Image", "DiffusionPipeline"]:
                NFO(f"Failed to detect type for {class_name} {list(sub_segments)}\n")
            else:
                mir_prefix = "info." + mir_prefix
        if class_name == "StableDiffusion3InpaintPipeline" or repo_path in ["stabilityai/stable-diffusion-3-medium-diffusers"]:
            class_name = "StableDiffusion3Pipeline"
            repo_path = "stabilityai/stable-diffusion-3.5-medium"
        if class_name == "HunyuanVideoFramepackPipeline" or repo_path in ["hunyuanvideo-community/HunyuanVideo"]:
            class_name = "HunyuanVideoPipeline"
        mir_series, mir_comp = list(tag_model_from_repo(repo_path, decoder))
        mir_series = mir_prefix + "." + mir_series
        repo_path = migrations(repo_path)
        # modalities = add_mode_types(mir_tag=[mir_series, mir_comp])
        prefixed_data = {
            "repo": repo_path,
            "pkg": {0: {"diffusers": class_name}},
            # "mode": modalities.get("mode"),
        }
        return mir_series, {mir_comp: prefixed_data}


def tag_pipe(repo_path: str, class_name: str, addendum: dict) -> tuple:
    """Convert model repo pipes to MIR tags, classifying by feature\n
    :param name: Repo path
    :param class_name: The HF Diffusers class for the model
    :return: A segmented MIR tag useful for appending index entries"""
    mir_series, mir_data = create_pipe_entry(repo_path=repo_path, class_name=class_name)
    mir_prefix, mir_series = mir_series.rsplit(".", 1)
    mir_comp = list(mir_data)[0]
    return mir_prefix, mir_series, {mir_comp: addendum}


def find_diffusers_docstrings() -> Generator[list[DocStringEntry]]:
    """Pull down docstrings from 🤗Diffusers pipelines, minimizing internet requests\n
    :return: Docstrings for common diffusers models"""
    import diffusers.pipelines as diffusers_pipelines

    docstring_patterns = EXCLUSIONS
    exclusion_list = docstring_patterns["exclusion_list"]
    uncommon_naming = docstring_patterns["uncommon_naming"]
    for pipe_name in IMPORT_STRUCTURE.keys():
        if pipe_name not in exclusion_list:
            file_specific = uncommon_naming.get(pipe_name, pipe_name)
            if import_name := getattr(diffusers_pipelines, str(pipe_name)):
                file_names = list(getattr(import_name, "_import_structure", {}).keys()) or [f"pipeline_{file_specific}"]
                yield list(retrieve_diffusers_docstrings(pipe_name, file_names))
            else:
                continue


def diffusers_index() -> dict[str, dict[str, dict[str, Any]]]:
    """Generate diffusion model data for MIR index\n
    :return: Dictionary ready to be applied to MIR data fields
    """
    special_repos = {
        "black-forest-labs/FLUX.1-schnell": "black-forest-labs/FLUX.1-dev",
        # "stabilityai/stable-diffusion-3-medium-diffusers": "stabilityai/stable-diffusion-3.5-medium",
    }
    special_classes = {
        # "StableDiffusion3Pipeline": "stabilityai/stable-diffusion-3.5-medium",  # NOT sd3
        "HunyuanDiTPipeline": "tencent-hunyuan/hunyuandiT-v1.2-diffusers",  #  NOT hyd .ckpt
        "ChromaPipeline": "lodestones/Chroma",
    }
    for class_name, swap_repo in special_classes.items():
        if parsed_data.pipe_class == class_name:
            parsed_data.pipe_repo = swap_repo
    extracted_docstrings = find_diffusers_docstrings()
    model_info = [extract for pipeline in extracted_docstrings for extract in pipeline]
    pipe_data = {}  # pipeline_stable_diffusion_xl_inpaint

    for extracted in model_info:
        parsed_data: DocParseData = parse_docs(extracted.doc_string)
        if parsed_data is None:
            print(f"Doc string not found in '{extracted.package_name}' in {extracted.file_name}")
            continue
        for class_name, swap_repo in special_classes.items():
            if parsed_data.pipe_class == class_name:
                parsed_data.pipe_repo = swap_repo
                break
        model_class_obj = import_object_named(parsed_data.pipe_class, extracted.pipe_module.__name__)
        if not model_class_obj:
            continue
        try:
            series, comp_data = create_pipe_entry(parsed_data.pipe_repo, parsed_data.pipe_class)
        except TypeError:
            pass  # Attempt 1
        if pipe_data.get(series):
            if "img2img" in parsed_data.pipe_class.lower():
                continue
        pipe_data.setdefault(series, {}).update(comp_data)
        special_conditions = special_repos | special_classes
        if parsed_data.staged_class or parsed_data.pipe_repo in list(special_conditions):
            test = special_conditions.get(parsed_data.pipe_repo)
            if test:
                staged_repo = test
                parsed_data.staged_class = parsed_data.pipe_class
            try:
                series, comp_data = create_pipe_entry(
                    staged_repo if parsed_data.staged_repo else parsed_data.pipe_repo,
                    parsed_data.staged_class  #
                    if parsed_data.staged_class
                    else parsed_data.pipe_class,
                )
            except TypeError as error_log:
                NFO(series, comp_data)
                NFO(error_log)
                continue  # Attempt 2,
            pipe_data.setdefault(series, {}).update(comp_data)
    return dict(pipe_data)


# def pull_weight_map(repo_id: str, arch: str) -> Dict[str, str]:
#     from nnll.download.hub_cache import download_hub_file

#     model_file = download_hub_file(
#         repo_id=f"{repo_id}/tree/main/{arch}",
#         source="huggingface",
#         file_name="diffusion_pytorch_model.safetensors.index.json",
#         local_dir=".tmp",
#     )


# @MODE_DATA.decorator
# def add_mode_types(mir_tag: list[str], data: dict | None = None) -> dict[str, list[str] | str]:
#     """_summary_\n
#     :param mir_tag: _description_
#     :param data: _description_, defaults to None
#     :return: _description_"""
#     fused_tag = ".".join(mir_tag)

#     mir_details = {
#         "mode": data.get(fused_tag, {}).get("pipeline_tag"),
#         "pkg_type": data.get(fused_tag, {}).get("library_type"),
#         "tags": data.get(fused_tag, {}).get("tags"),
#     }
#     return mir_details
