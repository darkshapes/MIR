### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""

from typing import Any, Callable, Dict, List, Tuple
import os
import sys
from nnll.monitor.file import nfo, dbug

if "pytest" in sys.modules:
    import diffusers

# from nnll.metadata.helpers import snake_caseify
from nnll.tensor_pipe.deconstructors import scrape_docs, cut_docs, root_class


def mir_label(mir_prefix: str, repo_path: str) -> Tuple[str]:
    """Create a mir label from a repo path\n
    :param mir_prefix: Known period-separated prefix and model type
    :param repo_path: Typical remote source repo path, A URL without domain
    :return: The assembled mir tag with compatibility pre-separated
    """
    import re

    name = os.path.basename(repo_path).lower().replace("_", "-").replace(".", "-")
    patterns = [
        r"-\d{3,}$",  # "-" and 3 digits
        r"-\d{4,}[px].*",  # "-" and 4 digits
        r"-\d{4,}.*",  # "-" and 4 digits
        r"-\d{1,2}[bBmM]$",  # "-" one or two digit number and "b" or "B" parameter model
        r"-v\d{1,2}$",  # "v" followed by one or two digits
        r"-dev$",
        r"-large$",
        r"-medium$",
        r"-schnell$",
        r"-prior$",
        r"-full$",
        r"-xt$",
        r"-box$",
        r"-preview$",
        r"-diffusers$",
        r"-base.*",
    ]

    for pattern in patterns:
        compiled = re.compile(pattern)
        if re.search(compiled, name):
            within = re.search(compiled, name).group()
            if within:
                suffix = within.strip("-")
                split_name = name.split(within)
                return mir_prefix + split_name[0], suffix

    return mir_prefix + name, "base"


def mir_index() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Generate diffusion model data for mir index\n
    :return: Dictionary ready to be applied to mir data fields
    """
    special_cases = {
        "black-forest-labs/FLUX.1-schnell": "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-3.5-medium": "stabilityai/stable-diffusion-3.5-large",
    }
    pipes = [scrape_docs(docstring) for docstring in list(cut_docs())]
    pipe_data = {}
    for entry in pipes:
        pipe_class, repo_path, staged_class, staged_repo = entry
        if pipe_class == "StableDiffusion3Pipeline":
            repo_path = "stabilityai/stable-diffusion-3.5-medium"
        elif pipe_class == "tencent-hunyuan/hunyuandit-diffusers":
            repo_path = "tencent-hunyuan/hunyuandiT-v1.2-diffusers"
        pipe_class = pipe_class
        series, comp_data = create_pipe_entry(repo_path, pipe_class)
        pipe_data.setdefault(series, {}).update(comp_data)  # Store as a list to avoid overwriting
        if staged_repo or any(repo_path in i for i in special_cases):
            test = special_cases.get(repo_path)
            if test:
                staged_repo = test
                staged_class = pipe_class
            series, comp_data = create_pipe_entry(staged_repo, staged_class)
            pipe_data.setdefault(series, {}).update(comp_data)  # Append instead of overwrite
    return {k: v for k, v in pipe_data.items()}


def create_pipe_entry(repo_path, pipe_class):
    import diffusers  # pylint:disable=redefined-outer-name

    if not repo_path and pipe_class:
        raise TypeError(f"'repo_path' {repo_path} or 'pipe_class' {pipe_class} unset")
    mir_prefix = "info."
    pipe_data = getattr(diffusers, pipe_class)
    sub_classes = root_class(pipe_data)
    if "unet" in sub_classes or "prior" in sub_classes or "decoder" in sub_classes or "kandinsky" in repo_path or "shap-e" in repo_path:
        mir_prefix = "info.unet."
    elif "transformer" in sub_classes:
        mir_prefix = "info.dit."
    mir_series, mir_comp = mir_label(mir_prefix, repo_path)
    prefixed_data = {
        "repo": repo_path,
        "pkg": {0: {"diffusers": pipe_class}},
    }
    if pipe_class == "FluxPipeline":
        pipe_class = {1: {"mflux": "Flux1"}}
        prefixed_data["pkg"].update(pipe_class)
    return mir_series, {mir_comp: prefixed_data}
