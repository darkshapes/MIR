# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Any
from mir import PARAMETERS, BREAKING, SEARCH


def tag_model_from_repo(repo_title: str, decoder=False, data: dict | None = None) -> tuple[str, Any]:
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

    pattern_2 = re.compile(PARAMETERS)
    clean_parts = [re.sub(pattern_2, "", segment.lower()) for segment in parts]
    cleaned_string = "-".join([x for x in clean_parts if x])
    cleaned_string = re.sub(subtraction_prefixes, "", cleaned_string)
    cleaned_string = re.sub("-it", "", cleaned_string.replace("-bit", "")).replace("--", "-")
    cleaned_string = cleaned_string.replace("-b-", "")
    # print(cleaned_string)
    suffix_match = re.findall(BREAKING, cleaned_string)  # Check for breaking suffixes first
    if suffix_match:
        suffix = next(iter(suffix for suffix in suffix_match[0] if suffix))
        cleaned_string = re.sub(suffix.lower(), "-", cleaned_string).rstrip("-,")
    else:
        suffix = root
    cleaned_string = re.sub(r"[._]+", "-", cleaned_string.lower()).strip("-_")
    return (cleaned_string, suffix)
