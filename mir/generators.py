### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""

from typing import Any, Dict, Tuple, Callable, List
import os
import sys
from nnll.monitor.file import dbug, nfo, dbuq
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
        r"-\d{1,2}[bBmMkK]",  # "-" one or two digit number and "b" or "B" parameter model
        r"-v\d{1,2}",  # "v" followed by one or two digits
        r"-\d{3,}$",  # "-" and 3 digits
        r"-\d{4,}.*",  # "-" and 4 digits
        r"-\d{4,}[px].*",  # "-" and 4 digits
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
                return mir_prefix + "." + split_name[0], suffix

    suffix = "decoder" if decoder else "base"
    return mir_prefix + "." + name, "base"


def diffusers_index() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Generate diffusion model data for MIR index\n
    :return: Dictionary ready to be applied to MIR data fields
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
    mir_prefix = "info"
    pipe_data = getattr(diffusers, pipe_class)
    sub_classes = root_class(pipe_data)
    decoder = "decoder" in sub_classes
    dbuq(pipe_class)
    dbuq(repo_path)
    if repo_path in ["openai/shap-e", "kandinsky-community/kandinsky-3"]:
        mir_prefix = "info.unet"
    else:
        mir_prefix = flag_config(**sub_classes)
    mir_series, mir_comp = mir_label(mir_prefix, repo_path, decoder)
    prefixed_data = {
        "repo": repo_path,
        "pkg": {0: {"diffusers": pipe_class}},
    }
    if pipe_class == "FluxPipeline":
        pipe_class = {1: {"mflux": "Flux1"}}
        prefixed_data["pkg"].update(pipe_class)
    return mir_series, {mir_comp: prefixed_data}


def flag_config(transformers: bool = False, **kwargs):
    """Set type of MIR prefix depending on model type\n
    :param transformers: Use transformers data instead of diffusers data, defaults to False
    :raises ValueError: Model type not detected
    :return: _description_"""
    xfmr_flags = {
        "info.detr": ("use_timm_backbone", "_resnet_"),
        "info.cnn": ("bbox_cost",),
        "info.rnn": ("lru_width",),
        "info.gan": ("codebook_dim", "kernel_size", "kernel_predictor_conv_size"),
        "info.mamba": (
            "mamba_expand",
            "parallel_attn",
        ),
        "info.vit": (
            "use_swiglu_ffn",
            "projection_dim",
            "vlm_config",
            "crop_size",
            "out_indices",
            "logit_scale_init_value",
            "image_size",
            "vision_config",
            "hidden_sizes",
            "image_token_id",
        ),
        "info.autoencoder": ("classifier_proj_size", "position_embedding_type", "separate_cls", "keypoint_detector_config", "local_attention"),
        "info.transformer": (
            "encoder_attention_heads",
            "encoder_layers",
            "decoder_layers",
            "decoder_hidden_size",
            "encoder_hidden_size",
            "is_encoder_decoder",
            "encoder_config",
            "audio_token_index",
        ),
        "info.autoregressive": ("ffn_dim", "num_codebooks", "vq_config", "attn_config", "n_head", "rms_norm_eps", "rope_theta", "head_dim", "hidden_dropout_prob"),
    }
    diff_flags = {
        "info.unet": ("unet", "prior", "decoder"),
        "info.dit": ("transformer",),
    }

    if transformers:
        flags = xfmr_flags
    else:
        flags = diff_flags
    for mir_prefix, key_match in flags.items():
        if any(kwargs.get(param) for param in key_match):
            return mir_prefix
    nfo("Unrecognized model type")
    dbuq("Unrecognized model type")


def transformers_index():
    """Generate LLM model data for MIR index\n
    :return: Dictionary ready to be applied to MIR data fields"""

    import re
    import transformers
    from nnll.tensor_pipe.deconstructors import stock_llm_data

    corrections = {
        "GraniteSpeechForConditionalGeneration": {
            "repo_path": "ibm-granite/granite-speech-3.3-8b",
            "sub_segments": {"encoder_layers": [""], "decoder_layers": [""]},
        },
        "GraniteModel": {
            "repo_path": "ibm-granite/granite-3.3-2b-base",
            "sub_segments": {"rope_theta": [""]},
        },
        "DPRQuestionEncoder": {
            "repo_path": "facebook/dpr-question_encoder-single-nq-base",
            "sub_segments": {"local_attention": [""], "classifier_proj_size": [""]},
        },
        "CohereModel": {
            "repo_path": "CohereForAI/c4ai-command-r-v01",
            "sub_segments": {"attn_config": [""], "num_codebooks": [""]},
        },
        "Cohere2Model": {
            "repo_path": "CohereLabs/c4ai-command-r7b-12-2024",
            "sub_segments": {"attn_config": [""], "num_codebooks": [""]},
        },
        "GraniteMoeHybridModel": {
            "repo_path": "ibm-research/PowerMoE-3b",
        },
        "GraniteMoeModel": {
            "repo_path": "ibm-research/PowerMoE-3b",
        },
        "AriaModel": {
            "repo_path": "rhymes-ai/Aria-Chat",
            "sub_segments": {"vision_config": [""], "text_config": [""]},
        },
        "TimmWrapperModel": {
            "repo_path": "timm/resnet18.a1_in1k",
            "sub_segments": {
                "_resnet_": [""],
            },
        },
    }

    mir_data = {}
    transformers_data: Dict[Callable, List[str]] = stock_llm_data()
    for model_class_obj, model_data in transformers_data.items():
        class_name = model_class_obj.__name__
        dbuq(class_name)
        if class_name in list(corrections):  # these are corrected because `root_class` doesn't return anything in these cases
            repo_path = corrections[class_name]["repo_path"]
            sub_segments = corrections[class_name].get("sub_segments", root_class(model_data["config"][-1], "transformers"))
            dbuq(repo_path)

        else:
            repo_path = ""
            doc_attempt = [getattr(transformers, model_data["config"][-1]), model_class_obj.forward]
            for pattern in doc_attempt:
                doc_string = pattern.__doc__
                matches = re.findall(r"\[([^\]]+)\]", doc_string)
                if matches:
                    repo_path = next(iter(snip.strip('"').strip() for snip in matches if "/" in snip))
                    repo_path
                    break
            sub_segments: Dict[str, List[str]] = root_class(model_data["config"][-1], "transformers")
        if sub_segments and list(sub_segments) != ["kwargs"] and list(sub_segments) != ["use_cache", "kwargs"] and repo_path is not None:
            dbuq(class_name)
            mir_prefix = flag_config(transformers=True, **sub_segments)
            if mir_prefix is None:
                nfo(f"Failed to detect type for {class_name} {list(sub_segments)}")
                dbuq(class_name, sub_segments, model_class_obj, model_data)
                continue
            mir_series, mir_comp = mir_label(mir_prefix, repo_path)
            mir_data.setdefault(
                mir_series,
                {
                    mir_comp: {
                        "repo": repo_path,
                        "pkg": {0: {"transformers": class_name}},
                    }
                },
            )
    return mir_data
