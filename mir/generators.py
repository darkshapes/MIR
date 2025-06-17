### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""

from typing import Any, Dict, Generator, List, Tuple
import os
import sys
from nnll.monitor.file import dbug
from nnll.tensor_pipe.deconstructors import scrape_docs, cut_docs, root_class

if "pytest" in sys.modules:
    import diffusers  # noqa # pyright:ignore[reportMissingImports] # pylint:disable=unused-import

# from nnll.metadata.helpers import snake_caseify


def mir_label(mir_prefix: str, repo_path: str, decoder=False) -> Tuple[str]:
    """Create a mir label from a repo path\n
    :param mir_prefix: Known period-separated prefix and model type
    :param repo_path: Typical remote source repo path, A URL without domain
    :return: The assembled mir tag with compatibility pre-separated
    """
    import re

    name = os.path.basename(repo_path).lower().replace("_", "-").replace("1.0", "").replace(".", "-")
    patterns = [
        r"-\d{3,}$",  # "-" and 3 digits
        r"-\d{4,}[px].*",  # "-" and 4 digits
        r"-\d{4,}.*",  # "-" and 4 digits
        r"-\d{1,2}[bBmM]$",  # "-" one or two digit number and "b" or "B" parameter model
        r"-v\d{1,2}",  # "v" followed by one or two digits
        r"-diffusers$",
        r"-large$",
        r"-medium$",
        r"-prior$",
        r"-full$",
        r"-xt$",
        r"-box$",
        r"-preview$",
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

    suffix = "decoder" if decoder else "base"
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
    dbug(pipes)
    pipe_data = {}
    for entry in pipes:
        pipe_class, repo_path, staged_class, staged_repo = entry
        if pipe_class == "StableDiffusion3Pipeline":
            repo_path = "stabilityai/stable-diffusion-3.5-medium"  # to avoid 3 and use 3.5
        elif pipe_class == "HunyuanDiTPipeline":
            repo_path = "tencent-hunyuan/hunyuandiT-v1.2-diffusers"  # to avoid 1 and use 1.2
        elif pipe_class == "ChromaPipeline":
            repo_path = "lodestones/Chroma"
        pipe_class = pipe_class.strip('"')
        series, comp_data = create_pipe_entry(repo_path, pipe_class)
        pipe_data.setdefault(series, {}).update(comp_data)  # update empty dict, preventing rewrite of others in series
        if staged_repo or any(repo_path in case for case in special_cases):  # these share the same pipe, (missing their own docstring)
            test = special_cases.get(repo_path)
            if test:
                staged_repo = test
                staged_class = pipe_class
            series, comp_data = create_pipe_entry(staged_repo, staged_class)
            pipe_data.setdefault(series, {}).update(comp_data)  # Update empty dict rather than entire series
    return dict(pipe_data)


def create_pipe_entry(repo_path: str, pipe_class: str) -> tuple[str, Dict[str, Dict[Any, Any]]]:
    """Create a pipeline article and generate corresponding information according to the provided repo path and pipeline category\n
    :param Repo_path (str): Repository path.
    :param Pipe_class (str): pipe class name.
    :raises TypeError: If 'repo_path' or 'pipe_class' are not set.
    :return: Tuple: The data structure containing mir_series and mir_comp is used for subsequent processing.
    """
    import diffusers  # pyright: ignore[reportMissingImports] # pylint:disable=redefined-outer-name

    if not repo_path and pipe_class:
        raise TypeError(f"'repo_path' {repo_path} or 'pipe_class' {pipe_class} unset")
    mir_prefix = "info."
    pipe_data = getattr(diffusers, pipe_class)
    sub_classes = root_class(pipe_data)
    decoder = "decoder" in sub_classes
    if "unet" in sub_classes or "prior" in sub_classes or decoder or "kandinsky" in repo_path or "shap-e" in repo_path:
        mir_prefix = "info.unet."
    elif "transformer" in sub_classes:
        mir_prefix = "info.dit."
    mir_series, mir_comp = mir_label(mir_prefix, repo_path, decoder)
    prefixed_data = {
        "repo": repo_path,
        "pkg": {0: {"diffusers": pipe_class}},
    }
    if pipe_class == "FluxPipeline":
        pipe_class = {1: {"mflux": "Flux1"}}
        prefixed_data["pkg"].update(pipe_class)
    return mir_series, {mir_comp: prefixed_data}
