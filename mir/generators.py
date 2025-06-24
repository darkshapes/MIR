### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""

from typing import Any, Dict, Tuple, Callable, List, Optional, Generator, Union
import os
import sys

from mir.json_cache import JSONCache, TEMPLATE_PATH_NAMED  # pylint:disable=no-name-in-module
from logging import Logger, INFO

nfo_obj = Logger(INFO)
nfo = nfo_obj.info


def root_class(module: Union[Callable, str], pkg_name: Optional[str] = None) -> Dict[str, List[str]]:
    """Pick apart a Diffusers or Transformers pipeline class and find its constituent parts\n
    :param module: Origin pipeline as a class or as a string
    :param library: name of a library to import the class from, only if a string is provided
    :return: Dictionary of sub-classes from the `module`"""

    import inspect
    from importlib import import_module

    if pkg_name and isinstance(module, str):
        module = import_module(module, pkg_name)
    signature = inspect.signature(module.__init__)
    class_names = {}
    for folder, param in signature.parameters.items():
        if folder != "self":
            sub_module = str(param.annotation).split("'")
            if len(sub_module) > 1 and sub_module[1] not in [
                "bool",
                "int",
                "float",
                "complex",
                "str",
                "list",
                "tuple",
                "dict",
                "set",
            ]:
                class_names.setdefault(folder, sub_module[1].split("."))
    return class_names


def scrape_docs(doc_string: str) -> Tuple[str,]:
    """Eat the 🤗Diffusers docstrings as a treat, leaving any tasty repo and class morsels neatly arranged as a dictionary.\n
    Nom.
    :param doc_string: String literal from library describing the class
    :return: A yummy dictionary of relevant class and repo strings"""

    pipe_prefix = [">>> adapter = ", ">>> pipe_prior = ", ">>> pipe = ", ">>> pipeline = ", ">>> blip_diffusion_pipe = ", ">>> gen_pipe = ", ">>> prior_pipe = "]
    repo_prefixes = ["repo_id", "model_ckpt", "model_id_or_path", "model_id", "repo"]
    class_method = [".from_pretrained(", ".from_single_file("]
    staged_class_method = ".from_pretrain("
    staged = None
    staged_class = None
    staged_repo = None
    joined_docstring = " ".join(doc_string.splitlines())

    for prefix in pipe_prefix:
        pipe_doc = joined_docstring.partition(prefix)[2]  # get the string segment that follows pipe assignment
        if prefix == pipe_prefix[-2]:  # continue until loop end [exhaust last two items in list above]
            staged = pipe_doc
        elif pipe_doc and not staged:
            break
    for method_name in class_method:
        if method_name in pipe_doc:
            pipe_class = pipe_doc.partition(method_name)[0]  # get the segment preceding the class' method call
            repo_path = pipe_doc.partition(method_name)  # break segment at method
            repo_path = repo_path[2].partition(")")[0]  # segment after is either a repo path or a reference to it, capture the part before the parenthesis
            repo_path = repo_path.replace("...", "").strip()  # remove any ellipsis and empty space
            repo_path = repo_path.partition('",')[0]  # partition at commas, repo is always the first argument
            repo_path = repo_path.strip('"')  # strip remaining quotes
            # * the star below could go here?
            if staged:
                staged_class = staged.partition(staged_class_method)[0]  # repeat with any secondary stages
                staged_repo = staged.partition(staged_class_method)
                staged_repo = staged_repo[2].partition(")")[0]
                staged_repo = staged_repo.replace("...", "").strip()
                staged_repo = staged_repo.partition('",')[0]
                staged_repo = staged_repo.strip('"')
            break
        else:
            continue
    for prefix in repo_prefixes:  # * this could move up
        if prefix in repo_path and not staged:  # if  don't have the repo path, but only a reference
            repo_variable = f"{prefix} = "  # find the variable assignment
            repo_path = next(line.partition(repo_variable)[2].split('",')[0] for line in doc_string.splitlines() if repo_variable in line).strip('"')
            break
    return pipe_class, repo_path, staged_class, staged_repo


def cut_docs() -> Generator:
    """Draw down docstrings from 🤗Diffusers library, minimizing internet requests\n
    :return: Docstrings for common diffusers models"""

    import pkgutil
    from importlib import import_module
    import diffusers.pipelines

    non_standard = {
        "cogvideo": "cogvideox",
        "cogview3": "cogview3plus",
        "deepfloyd_if": "if",
        "cosmos": "cosmos2_text2image",  # search folder for all files containing 'EXAMPLE DOC STRING'
        "visualcloze": "visualcloze_generation",
    }

    exclusion_list = [  # task specific, adapter, or no doc string. all can be be gathered by other means
        # these will be handled eventually
        "animatediff",  # adapter
        "controlnet",
        "controlnet_hunyuandit",  #: "hunyuandit_controlnet",
        "controlnet_xs",
        "controlnetxs",
        "controlnet_hunyuandit",
        "controlnet_sd3",
        "pag",  #
        "stable_diffusion_3_controlnet",
        "stable_diffusion_attend_and_excite",
        "stable_diffusion_sag",  #
        "t2i_adapter",
        "ledits_pp",  # "leditspp_stable_diffusion",
        "latent_consistency_models",  # "latent_consistency_text2img",
        "unclip",
        # these are uncommon afaik
        "dance_diffusion",  # no doc_string
        "dit",
        "ddim",
        "ddpm",
        "deprecated",
        "latent_diffusion",  # no doc_string
        "marigold",  # specific processing routines
        "omnigen",  # tries to import torchvision
        "paint_by_example",  # no docstring
        "pia",  # lora adapter
        "semantic_stable_diffusion",  # no_docstring
        "stable_diffusion_diffedit",
        "stable_diffusion_k_diffusion",  # tries to import k_diffusion
        "stable_diffusion_panorama",
        "stable_diffusion_safe",  # impossibru
        "text_to_video_synthesis",
        "unidiffuser",
    ]

    for _, name, is_pkg in pkgutil.iter_modules(diffusers.pipelines.__path__):
        if is_pkg and name not in exclusion_list:
            if name in non_standard:
                file_specific = non_standard[name]
            else:
                file_specific = name
            file_name = f"pipeline_{file_specific}"
            try:
                pipe_file = import_module(f"diffusers.pipelines.{name}.{file_name}")
            except ModuleNotFoundError:  # as error_log:
                nfo(f"Module Not Found for {name}")
                # dbug(error_log)
                pipe_file = None
            try:
                if pipe_file:
                    yield pipe_file.EXAMPLE_DOC_STRING
            except AttributeError:  # as error_log:
                nfo(f"Doc String Not Found for {name}")
                # dbug(error_log)
                # print(sub_classes)


TEMPLATE_FILE = JSONCache(TEMPLATE_PATH_NAMED)
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
    # dbug(pipes)
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


def create_pipe_entry(repo_path: str, class_name: str) -> tuple[str, Dict[str, Dict[Any, Any]]]:
    """Create a pipeline article and generate corresponding information according to the provided repo path and pipeline category\n
    :param Repo_path (str): Repository path.
    :param class_name (str): pipe class name.
    :raises TypeError: If 'repo_path' or 'class_name' are not set.
    :return: Tuple: The data structure containing mir_series and mir_comp is used for subsequent processing.
    """
    import diffusers  # pyright: ignore[reportMissingImports] # pylint:disable=redefined-outer-name

    mir_prefix = "info"
    model_class_obj = getattr(diffusers, class_name)
    sub_segments = root_class(model_class_obj)
    decoder = "decoder" in sub_segments
    # dbuq(class_name)
    # dbuq(repo_path)
    if repo_path in ["openai/shap-e", "kandinsky-community/kandinsky-3"]:
        mir_prefix = "info.unet"
    else:
        mir_prefix = flag_config(**sub_segments)
        if mir_prefix is None:
            nfo(f"Failed to detect type for {class_name} {list(sub_segments)}")
            # dbuq(class_name, sub_segments, model_class_obj)
            # return None
        else:
            mir_prefix = "info." + mir_prefix
    mir_series, mir_comp = mir_label(mir_prefix, repo_path, decoder)
    prefixed_data = {
        "repo": repo_path,
        "pkg": {0: {"diffusers": class_name}},
    }
    if class_name == "FluxPipeline":
        class_name = {1: {"mflux": "Flux1"}}
        prefixed_data["pkg"].update(class_name)
    return mir_series, {mir_comp: prefixed_data}


def flag_config(transformers: bool = False, **kwargs):
    """Set type of MIR prefix depending on model type\n
    :param transformers: Use transformers data instead of diffusers data, defaults to False
    :raises ValueError: Model type not detected
    :return: _description_"""

    @TEMPLATE_FILE.decorator
    def _read_data(data: Optional[Dict[str, str]] = None):
        return data

    template_data = _read_data()

    if transformers:
        flags = template_data["arch"]["xfmr"]  # pylint:disable=unsubscriptable-object
    else:
        flags = template_data["arch"]["diff"]  # pylint:disable=unsubscriptable-object
    for mir_prefix, key_match in flags.items():
        if any(kwargs.get(param) for param in key_match):
            return mir_prefix
    nfo("Unrecognized model type")
    # dbuq("Unrecognized model type")


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
        # dbuq(class_name)
        if class_name in list(corrections):  # these are corrected because `root_class` doesn't return anything in these cases
            repo_path = corrections[class_name]["repo_path"]
            sub_segments = corrections[class_name].get("sub_segments", root_class(model_data["config"][-1], "transformers"))
            # dbuq(repo_path)

        else:
            repo_path = ""
            doc_attempt = [getattr(transformers, model_data["config"][-1]), model_class_obj.forward]
            for pattern in doc_attempt:
                doc_string = pattern.__doc__
                matches = re.findall(r"\[([^\]]+)\]", doc_string)
                if matches:
                    repo_path = next(iter(snip.strip('"').strip() for snip in matches if "/" in snip))
                    break
            sub_segments: Dict[str, List[str]] = root_class(model_data["config"][-1], "transformers")
        if sub_segments and list(sub_segments) != ["kwargs"] and list(sub_segments) != ["use_cache", "kwargs"] and repo_path is not None:
            # dbuq(class_name)
            mir_prefix = flag_config(transformers=True, **sub_segments)
            if mir_prefix is None:
                nfo(f"Failed to detect type for {class_name} {list(sub_segments)}")
                # dbuq(class_name, sub_segments, model_class_obj, model_data)
                continue
            else:
                mir_prefix = "info." + mir_prefix
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
