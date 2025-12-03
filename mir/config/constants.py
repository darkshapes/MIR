# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
from typing import List, Optional, Union
from mir.config.json_io import read_json_file
import os
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_MAPPING_NAMES,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.modeling_utils import PretrainedConfig


def generate_supported_model_class_names(
    model_name: type[PretrainedConfig],
    supported_tasks: Optional[Union[str, list[str]]] = None,
) -> list[str]:
    task_mapping = {
        "default": MODEL_MAPPING_NAMES,
        "pretraining": MODEL_FOR_PRETRAINING_MAPPING_NAMES,
        "next-sentence-prediction": MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
        "masked-lm": MODEL_FOR_MASKED_LM_MAPPING_NAMES,
        "causal-lm": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        "seq2seq-lm": MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        "speech-seq2seq": MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
        "multiple-choice": MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
        "document-question-answering": MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
        "question-answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
        "sequence-classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
        "token-classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
        "masked-image-modeling": MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
        "image-classification": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        "zero-shot-image-classification": MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        "ctc": MODEL_FOR_CTC_MAPPING_NAMES,
        "audio-classification": MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
        "semantic-segmentation": MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
        "backbone": MODEL_FOR_BACKBONE_MAPPING_NAMES,
        "image-feature-extraction": MODEL_FOR_IMAGE_MAPPING_NAMES,
        "video-classification": MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
    }

    if supported_tasks is None:
        supported_tasks = task_mapping.keys()
    if isinstance(supported_tasks, str):
        supported_tasks = [supported_tasks]

    model_class_names = []
    for task in supported_tasks:
        class_name = task_mapping[task].get(model_name, None)
        if class_name:
            model_class_names.append(class_name)

    return model_class_names


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
versions = read_json_file(os.path.join(os.path.dirname(__file__), "versions.json"))
template = read_json_file(os.path.join(os.path.dirname(__file__), "template.json"))

MIR_PATH_NAMED = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mir.json")

BREAKING_SUFFIX = r".*(?:-)(prior)$|.*(?:-)(diffusers)$|.*[_-](\d{3,4}px|-T2V$|-I2V$)"
PARAMETERS_SUFFIX = r"(\d{1,4}[KkMmBb]|[._-]\d+[\._-]\d+[Bb][._-]).*?$"
SEARCH_SUFFIX = r"\d+[._-]?\d+[BbMmKk](it)?|[._-]\d+[BbMmKk](it)?"
