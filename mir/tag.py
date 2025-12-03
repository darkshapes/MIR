# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import List
from mir.config.constants import PARAMETERS_SUFFIX, BREAKING_SUFFIX


def make_mir_tag(repo_title: str, decoder=False, data: dict = None) -> List[str]:
    """Create a mir label from a repo path\n
    :param mir_prefix: Known period-separated prefix and model type
    :param repo_path: Typical remote source repo path, A URL without domain
    :return: The assembled mir tag with compatibility pre-separated"""
    import re

    # print(repo_title)

    root = "decoder" if decoder else "*"
    repo_title = repo_title.split(":latest")[0]
    repo_title = repo_title.split(":Q")[0]
    repo_title = repo_title.split(r"/")[-1].lower()
    pattern = r"^.*[v]?(\d{1}+\.\d).*"
    match = re.findall(pattern, repo_title)
    if match:
        if next(iter(match)):
            repo_title = repo_title.replace(next(iter(match))[-1], "")
    parts = repo_title.replace(".", "").split("-")
    if len(parts) == 1:
        parts = repo_title.split("_")
    subtraction_prefixes = r"\d.b-|\-rl|tiny|large|mlx|onnx|gguf|medium|base|multimodal|mini|instruct|full|:latest|preview|small|pro|beta|hybrid|plus|dpo|community"

    pattern_2 = re.compile(PARAMETERS_SUFFIX)
    clean_parts = [re.sub(pattern_2, "", segment.lower()) for segment in parts]
    cleaned_string = "-".join([x for x in clean_parts if x])
    cleaned_string = re.sub(subtraction_prefixes, "", cleaned_string)
    cleaned_string = re.sub("-it", "", cleaned_string.replace("-bit", "")).replace("--", "-")
    cleaned_string = cleaned_string.replace("-b-", "")
    # print(cleaned_string)
    suffix_match = re.findall(BREAKING_SUFFIX, cleaned_string)  # Check for breaking suffixes first
    if suffix_match:
        suffix = next(iter(suffix for suffix in suffix_match[0] if suffix))
        cleaned_string = re.sub(suffix.lower(), "-", cleaned_string).rstrip("-,")
    else:
        suffix = root
    cleaned_string = re.sub(r"[._]+", "-", cleaned_string.lower()).strip("-_")
    return (cleaned_string, suffix)


def make_scheduler_tag(series_name: str) -> tuple[str]:
    """Create a mir label from a scheduler operation\n
    :param class_name: Known period-separated prefix and model type
    :return: The assembled mir tag with compatibility pre-separated"""

    import re

    comp_name = None
    patterns = [r"Schedulers", r"Multistep", r"Solver", r"Discrete", r"Scheduler"]
    for scheduler in patterns:
        compiled = re.compile(scheduler)
        match = re.search(compiled, series_name)
        if match:
            comp_name = match.group()
            comp_name = comp_name.lower()
            break
    for pattern in patterns:
        series_name = re.sub(pattern, "", series_name)
    series_name.lower()
    # if not comp_name:
    #     comp_name = "*"
    return series_name, comp_name


def tag_base_model(repo_path: str, class_name: str, addendum: dict | None = None) -> tuple[str]:
    """Convert model repo paths to MIR tags, classifying by feature\n
    :param name: Repo path
    :param class_name: The HF transformers class for the model
    :return: A segmented MIR tag useful for appending index entries"""

    from mir.inspect.classes import extract_init_params
    from mir.indexers import flag_config

    annotations = extract_init_params(class_name.replace("Model", "Config"), "transformers")
    mir_prefix = flag_config(transformers=True, **annotations)
    base_series, base_comp = make_mir_tag(repo_path)
    if not addendum:
        return mir_prefix, base_series, base_comp
    else:
        mir_prefix = f"info.{mir_prefix}"
    return mir_prefix, base_series, {base_comp: addendum}


def tag_pipe(repo_path: str, class_name: str, addendum: dict) -> tuple:
    """Convert model repo pipes to MIR tags, classifying by feature\n
    :param name: Repo path
    :param class_name: The HF Diffusers class for the model
    :return: A segmented MIR tag useful for appending index entries"""

    from mir.indexers import create_pipe_entry

    mir_series, mir_data = create_pipe_entry(repo_path=repo_path, class_name=class_name)
    mir_prefix, mir_series = mir_series.rsplit(".", 1)
    mir_comp = list(mir_data)[0]
    return mir_prefix, mir_series, {mir_comp: addendum}


# def tag_mlx_model(repo_path: str, class_name: str, addendum: dict) -> tuple[str]:
#     dev_series, dev_comp = make_mir_tag("black-forest-labs/FLUX.1-dev")
#     schnell_series, schnell_comp = make_mir_tag("black-forest-labs/FLUX.1-schnell")
#     series, comp = make_mir_tag(repo_path)
#     if class_name == "Flux1":
#         mir_prefix = "info.dit"
#         base_series = dev_series
#         mir_comp = series
#         return mir_prefix, base_series, {base_comp: addendum}
