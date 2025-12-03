# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
from typing import List, Optional, Union
from mir.config.json_io import read_json_file
import os

from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_MAPPING_NAMES
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
import transformers

def mapped_cls(model_identifier: str):
    """Get model class from identifier without calling huggingface_hub.
    
    :param model_identifier: Model identifier like "bert-base-uncased" or "gpt2"
    :return: Model class (e.g., BertModel, GPT2Model)
    """
    # Extract code name from model identifier (e.g., "bert-base-uncased" -> "bert")
    # Handle various formats: "bert-base-uncased", "gpt2", "microsoft/DialoGPT-medium"
    code_name = model_identifier.split("/")[-1].split("-")[0].lower()
    
    # Method 1: Direct lookup via MODEL_MAPPING_NAMES (simplest)
    model_class_name = MODEL_MAPPING_NAMES.get(code_name, None)

    
    # Method 2: Via config class lookup (matches _get_model_class behavior more closely)
    config_class_name = CONFIG_MAPPING_NAMES.get(code_name)
    if config_class_name:
        config_class = getattr(transformers, config_class_name, None)
        if config_class:
            # Look up in MODEL_MAPPING using config class
            model_class = MODEL_MAPPING.get(config_class, None)
            if model_class:
                if isinstance(model_class, tuple):
                    model_class = model_class[0]
                    return model_class
    
    # Fallback: try with normalized code name (handle underscores/dashes)
    normalized = code_name.replace("_", "-")
    if normalized != code_name:
        print(f"normalized: {normalized}")
        model_class_name = MODEL_MAPPING_NAMES.get(normalized,  None)
        if model_class_name:
            return getattr(transformers, model_class_name, None)
    if model_class_name:
        if isinstance(model_class_name, tuple):
            model_class_name = model_class_name[0]
        return getattr(transformers, model_class_name, None)

    return None


class DocParseData:
    pipe_class: str
    pipe_repo: str
    staged_class: Optional[str] = None
    staged_repo: Optional[str] = None

    def __init__(self, pipe_class, pipe_repo, staged_class=None, staged_repo=None):
        self.pipe_class: str = pipe_class
        self.pipe_repo: str = pipe_repo
        self.staged_class: str = staged_class
        self.staged_repo: str = staged_repo


class DocStringParserConstants:
    """Constants used by DocStringParser for parsing docstrings."""

    pipe_prefixes: List[str] = [
        ">>> motion_adapter = ",
        ">>> adapter = ",  # if this moves, also change motion_adapter check
        ">>> controlnet = ",
        ">>> pipe_prior = ",
        ">>> pipe = ",
        ">>> pipeline = ",
        ">>> blip_diffusion_pipe = ",
        ">>> prior_pipe = ",
        ">>> gen_pipe = ",
    ]
    repo_variables: List[str] = [
        "controlnet_model",
        "controlnet_id",
        "base_model",
        "model_id_or_path",
        "model_ckpt",
        "model_id",
        "repo_base",
        "repo",
        "motion_adapter_id",
    ]
    call_types: List[str] = [".from_pretrained(", ".from_single_file("]
    staged_call_types: List[str] = [
        ".from_pretrain(",
    ]


package_map = {
    "diffusers": ("_import_structure", "diffusers.pipelines"),
    "transformers": ("MODEL_MAPPING_NAMES", "transformers.models.auto.modeling_auto"),
}
root_path = os.path.join(os.getcwd(), "mir")
versions = read_json_file(os.path.join(root_path, "spec", "versions.json"))
template = read_json_file(os.path.join(root_path, "spec", "template.json"))
print(root_path)
MIR_PATH_NAMED = os.path.join(root_path, "mir.json")

BREAKING_SUFFIX = r".*(?:-)(prior)$|.*(?:-)(diffusers)$|.*[_-](\d{3,4}px|-T2V$|-I2V$)"
PARAMETERS_SUFFIX = r"(\d{1,4}[KkMmBb]|[._-]\d+[\._-]\d+[Bb][._-]).*?$"
SEARCH_SUFFIX = r"\d+[._-]?\d+[BbMmKk](it)?|[._-]\d+[BbMmKk](it)?"
