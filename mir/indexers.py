# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""
# pylint:disable=no-name-in-module

import sys
from typing import Any, Callable

from mir.config.console import nfo
from mir.config.constants import ClassMapEntry, extract_init_parameters
from mir.config.conversion import get_repo_from_class_map, import_submodules
from mir.doc_parser import parse_docs, DocParseData
from mir.tag import mir_prefix_from_forward_pass, mir_tag_from_config, tag_model_from_repo

if "pytest" in sys.modules:
    import diffusers  # noqa # pyright:ignore[reportMissingImports] # pylint:disable=unused-import


def check_migrations(repo_path: str):
    """Replaces old organization names in repository paths with new ones.\n
    :param repo_path: Original repository path containing old organization names
    :return: Updated repository path with new organization names"""
    import os

    from mir.config.json_io import read_json_file

    root_folder = os.path.dirname(__file__)
    migration_file = os.path.join(os.path.join(root_folder, "spec", "repo_migrations.json"))
    repo_migrations = read_json_file(migration_file)
    for old_name, new_name in repo_migrations.items():
        if old_name in repo_path:
            repo_path = repo_path.replace(old_name, new_name)
    return repo_path


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
        sub_segments = extract_init_parameters(model_class_obj, "diffusers")
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
            mir_prefix = mir_prefix_from_forward_pass(**sub_segments)
            if mir_prefix is None and class_name not in ["AutoPipelineForImage2Image", "DiffusionPipeline"]:
                nfo(f"Failed to detect type for {class_name} {list(sub_segments)}\n")
            else:
                mir_prefix = "info." + mir_prefix
        if class_name == "StableDiffusion3InpaintPipeline" or repo_path in ["stabilityai/stable-diffusion-3-medium-diffusers"]:
            class_name = "StableDiffusion3Pipeline"
            repo_path = "stabilityai/stable-diffusion-3.5-medium"
        if class_name == "HunyuanVideoFramepackPipeline" or repo_path in ["hunyuanvideo-community/HunyuanVideo"]:
            class_name = "HunyuanVideoPipeline"
        mir_series, mir_comp = list(tag_model_from_repo(repo_path, decoder))
        mir_series = mir_prefix + "." + mir_series
        repo_path = check_migrations(repo_path)
        # modalities = add_mode_types(mir_tag=[mir_series, mir_comp])
        prefixed_data = {
            "repo": repo_path,
            "pkg": {0: {"diffusers": class_name}},
            # "mode": modalities.get("mode"),
        }
        return mir_series, {mir_comp: prefixed_data}


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
    from mir.inspect.metadata import find_diffusers_docstrings

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
        model_class_obj = import_submodules(parsed_data.pipe_class, f"diffusers.pipelines.{extracted.package_name}.{extracted.file_name}")
        if not model_class_obj:
            continue
        extract_init_parameters(model_class_obj)
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
                nfo(series, comp_data)
                nfo(error_log)
                continue  # Attempt 2,
            pipe_data.setdefault(series, {}).update(comp_data)
    return dict(pipe_data)


def transformers_index():
    """Generate LLM model data for MIR index\n
    :return: Dictionary ready to be applied to MIR data fields"""

    import os

    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES

    from mir.config.json_io import read_json_file

    root_folder = os.path.dirname(__file__)
    params_file = os.path.join(os.path.join(root_folder, "spec", "missing_params.json"))
    missing_config_params = read_json_file(params_file)
    from mir.inspect.metadata import map_transformers_classes

    mir_data = {}
    transformers_data: list[ClassMapEntry] = map_transformers_classes()
    for entry in transformers_data:
        repo_path = get_repo_from_class_map(entry)
        if config := missing_config_params.get(entry.name, {}):
            entry.config_params = config.get("params", entry.config_params)
            if not repo_path or entry.name == "gpt_oss":
                repo_path = config["repo_path"]
        if not repo_path:
            raise ValueError(f"Unable to determine repo from {entry}")
        if entry.config_params:
            mir_series, mir_comp, mir_suffix = mir_tag_from_config(entry, repo_path)
            # modalities = add_mode_types(mir_tag=[mir_series, mir_comp])

            repo_path = check_migrations(repo_path)
            tk_pkg = {}
            tokenizer_classes = TOKENIZER_MAPPING_NAMES.get(entry.name)
            if isinstance(tokenizer_classes, str):
                tokenizer_classes = [tokenizer_classes]
            # mode = modalities.get("mode")
            if tokenizer_classes:
                index = 0
                for tokenizer in tokenizer_classes:
                    if tokenizer:
                        tokenizer_class = import_submodules(tokenizer, "transformers")
                        tk_pkg.setdefault(index, {"transformers": f"{tokenizer_class.__module__}.{tokenizer_class.__name__}"})
                        index += 1
                if tk_pkg:
                    mir_data.get("info.encoder.tokenizer", mir_data.setdefault("info.encoder.tokenizer", {})).update(
                        {
                            mir_suffix: {
                                "pkg": tk_pkg,
                            }
                        },
                    )
            mir_data.setdefault(
                mir_series,
                {
                    mir_comp: {
                        "repo": repo_path,
                        "pkg": {
                            0: {"transformers": entry.model_name},
                        },
                        # "mode": mode,
                    },
                },
            )
    return mir_data


def mlx_repo_capture(base_repo: str = "mlx-community"):
    import os
    import re

    try:
        import mlx_audio  # type: ignore
    except ImportError:
        return {}
    result = {}
    result_2 = {}
    folder_path_named: str = os.path.dirname(mlx_audio.__file__)
    for root, dir, file_names in os.walk(folder_path_named):
        for file in file_names:
            if file.endswith((".py", ".html", ".md", ".ts")):
                with open(os.path.join(root, file), "r") as open_file:
                    content = open_file.read()
                    if "mlx-community/" in content:
                        matches = re.findall(base_repo + r'/(.*?)"', content)
                        for match in matches:
                            result[match] = f"{base_repo}/{match}"
                            previous_data = content[content.index(match) - 75 : content.index(match)].replace(base_repo, "")
                            class_match = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)
                            if class_match:
                                result_2[match] = {f"{base_repo}/{match}": [*class_match]}
                            else:
                                if os.path.basename(root) in ["tts", "sts"]:
                                    folder_name = match.partition("-")[0]
                                    file_path = os.path.join(root, "models", folder_name, folder_name + ".py")
                                    if os.path.exists(file_path):
                                        with open(file_path, "r") as model_file:
                                            read_data = model_file.read()  # type: ignore  # noqa
                                            class_match = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)

    return result_2


# def mlx_repo_capture(base_repo: str = "mlx-community"):
#     import os
#     import re
#     import mlx_audio

#     result = {}
#     result_2 = {}
#     folder_path_named: str = os.path.dirname(mlx_audio.__file__)
#     for root, _, file_names in os.walk(folder_path_named):
#         for file in file_names:
#             if file.endswith((".py", ".html", ".md", ".ts")):
#                 with open(os.path.join(root, file), "r") as open_file:
#                     content = open_file.read()
#                     if "mlx-community/" in content:
#                         matches = re.findall(base_repo + r'/(.*?)"', content)
#                         for match in matches:
#                             print(file)
#                             result[match] = f"{base_repo}/{match}"
#                             previous_data = content[content.index(match) - 75 : content.index(match)].replace(base_repo, "")
#                             matches = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)
#                             if matches:
#                                 result_2[match] = {f"{base_repo}/{match}": [*matches]}
#                             else:
#                                 result_2[match] = {f"{base_repo}/{match}": None}
#     return result_2


# def mlx_audio_scrape(base_repo: str = "mlx-community"):
#     import os
#     import re
#     import mlx_audio

#     result = {}
#     result_2 = {}
#     folder_path_named: str = os.path.dirname(mlx_audio.__file__)
#     for root, _, file_names in os.walk(folder_path_named):
#         for file in file_names:
#             if file.endswith((".py",)):
#                 with open(os.path.join(root, file), "r") as open_file:
#                     content = open_file.read()
#                     if "mlx-community/" in content:
#                         matches = re.findall(base_repo + r'/(.*?)"', content)
#                         for match in matches:
#                             result[match] = f"{base_repo}/{match}"
#                             previous_data = content[content.index(match) - 75 : content.index(match)].replace(base_repo, "")
#                             matches = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)
#                             if len(matches) > 1:
#                                 result_2[match] = {f"{base_repo}/{match}": [*matches]}
#                             else:
#                                 if "nn.Module" in content:
#                                     previous_data = content[content.rindex("nn.Module") - 50 : content.rindex("nn.Module")]
#                                     matches = re.search(r"(\w+)\.", previous_data, re.MULTILINE)
#                                     result_2[match] = {f"{base_repo}/{match}": [*matches]}
#     return result_2


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
