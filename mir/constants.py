### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->### <!-- // /*  d a r k s h a p e s */ -->


from enum import Enum
from typing import Annotated, Callable, List, Optional, Union

from pydantic import BaseModel, Field

from nnll.monitor.file import dbuq, nfo
from nnll.configure.init_gpu import first_available
from mir.json_cache import JSONCache, CUETYPE_PATH_NAMED  # pylint:disable=no-name-in-module

CUETYPE_CONFIG = JSONCache(CUETYPE_PATH_NAMED)


@CUETYPE_CONFIG.decorator
def has_api(api_name: str, data: dict = None) -> bool:
    """Check available modules, try to import dynamically.
    True for successful import, else False

    :param api_name: Constant name for API
    :param _data: filled by config decorator, ignore, defaults to None
    :return: _description_
    """
    from json.decoder import JSONDecodeError

    def set_true(api_name):
        nfo(f"Available {api_name}")
        return True

    def set_false(api_name):
        dbuq("|Ignorable| Source unavailable:", f"{api_name}")
        return False

    def check_host(api_name) -> bool:
        import httpcore
        import httpx
        from urllib3.exceptions import NewConnectionError, MaxRetryError
        import requests

        dbuq(f"Response from API  {api_data}")
        try:
            if api_data.get("api_url", 0):
                request = requests.get(api_data.get("api_url"), timeout=(1, 1))
                if request is not None:
                    if hasattr(request, "reason") and request.reason == "OK":  # The curious case of Ollama
                        return set_true(api_name)
                    status = request.json()
                    if status.get("result") == "OK":
                        return set_true(api_name)
        except JSONDecodeError as error_log:
            dbuq(error_log)
            dbuq(f"json for ! {api_data}")
            dbuq(request.status_code)
            if request.ok:
                return set_true(api_name)
            try:
                request.raise_for_status()
            except requests.HTTPError() as _error_log:
                dbuq(_error_log)
        except (
            requests.exceptions.ConnectionError,
            httpcore.ConnectError,
            httpx.ConnectError,
            ConnectionRefusedError,
            MaxRetryError,
            NewConnectionError,
            TimeoutError,
            OSError,
            RuntimeError,
            ConnectionError,
        ) as error_log:
            dbuq(error_log)

    try:
        api_data = data.get(api_name, False)  # pylint: disable=unsubscriptable-object
    except JSONDecodeError as error_log:
        dbuq(error_log)
    if not api_data:
        api_data = {"module": api_name.lower()}

    if api_name == "LM_STUDIO":
        try:
            from lmstudio import APIConnectionError, APITimeoutError, APIStatusError, LMStudioWebsocketError

            try:
                return check_host(api_name)
            except (LMStudioWebsocketError, APIConnectionError, APITimeoutError, APIStatusError, JSONDecodeError) as error_log:
                dbuq(error_log)
        except (UnboundLocalError, ImportError, ModuleNotFoundError, JSONDecodeError) as error_log:
            dbuq(error_log)
    elif api_name in ["LLAMAFILE", "CORTEX"]:
        try:
            from openai import APIConnectionError, APIStatusError, APITimeoutError

            try:
                return check_host(api_name)
            except (APIConnectionError, APITimeoutError, APIStatusError, JSONDecodeError) as error_log:
                dbuq(error_log)
        except (UnboundLocalError, ImportError, ModuleNotFoundError, JSONDecodeError) as error_log:
            dbuq(error_log)
    elif api_name in ["OLLAMA", "KAGGLE"]:
        try:
            return check_host(api_name)
        except (UnboundLocalError, ImportError, ModuleNotFoundError, JSONDecodeError) as error_log:
            dbuq(error_log)
    else:
        try:
            __import__(api_data.get("module"))
            if api_name not in ["OLLAMA", "LM_STUDIO", "CORTEX", "LLAMAFILE", "VLLM", "KAGGLE"]:
                return set_true(api_name)
        except (UnboundLocalError, ImportError, ModuleNotFoundError):
            dbuq("|Ignorable| Source unavailable:", f"{api_name}")
            return set_false(api_name)
        return check_host(api_name)
    return set_false(api_name)


class BaseEnum(Enum):
    """Base class for available system packages\n"""

    @classmethod
    def show_all(cls) -> List:
        """Show all POSSIBLE API types of a given class"""
        return [x for x, y in CueType.__members__.items()]

    @classmethod
    def show_available(cls) -> bool:
        """Show all AVAILABLE API types of a given class"""
        return [library.value[1] for library in list(cls) if library.value[0] is True]

    @classmethod
    def check_type(cls, type_name: str, server: bool = False) -> bool:
        """Check for a SINGLE API availability"""
        type_name = type_name.upper()
        if hasattr(cls, type_name):
            available = next(iter(getattr(cls, type_name).value))
            if server and available:
                return has_api(type_name)
            return available


class CueType(BaseEnum):
    """Model Provider constants\n
    Caches and servers\n
    <NAME: (Availability, IMPORT_NAME)>"""

    # Dfferentiation of boolean conditions
    # GIVEN : The state of all provider modules & servers are marked at launch

    OLLAMA: tuple = (has_api("OLLAMA"), "OLLAMA")
    HUB: tuple = (has_api("HUB"), "HUB")
    LM_STUDIO: tuple = (has_api("LM_STUDIO"), "LM_STUDIO")
    CORTEX: tuple = (has_api("CORTEX"), "CORTEX")
    LLAMAFILE: tuple = (has_api("LLAMAFILE"), "LLAMAFILE")
    VLLM: tuple = (has_api("VLLM"), "VLLM")
    MLX_AUDIO: tuple = (has_api("MLX_AUDIO"), "MLX_AUDIO")
    KAGGLE: tuple = (has_api("KAGGLE"), "KAGGLE")


example_str = ("function_name", "import.function_name")


class PkgType(BaseEnum):
    """Package dependency constants\n
    Collected info from hub model tags and dependencies\n
    <NAME: (Availability, IMPORT_NAME, [Github repositories*]\n
    *if applicable, otherwise IMPORT_NAME is pip package\n
    NOTE: NAME is colloquial and does not always match IMPORT_NAME>"""

    AUDIOGEN: tuple = (has_api("AUDIOCRAFT"), "AUDIOCRAFT", ["exdysa/facebookresearch-audiocraft-revamp"])  # this fork supports mps
    BAGEL: tuple = (has_api("BAGEL"), "BAGEL", ["bytedance-seed/BAGEL"])
    BITSANDBYTES: tuple = (has_api("BITSANDBYTES"), "BITSANDBYTES", [])  # bitsandbytes-foundation/bitsandbytes
    DIFFUSERS: tuple = (has_api("DIFFUSERS"), "DIFFUSERS", [])
    EXLLAMAV2: tuple = (has_api("EXLLAMAV2"), "EXLLAMAV2", [])  # turboderp-org/exllamav2
    F_LITE: tuple = (has_api("F_LITE"), "F_LITE", ["fal-ai/f-lite"])
    HIDIFFUSION: tuple = (has_api("HIDIFFUSION"), "HIDIFFUSION", ["megvii-research/HiDiffusion"])
    LUMINA_MGPT: tuple = (has_api("INFERENCE_SOLVER"), "INFERENCE_SOLVER", "Alpha-VLLM/Lumina-mGPT")
    MFLUX: tuple = (has_api("MFLUX"), "MFLUX", [])  # "filipstrand/mflux"
    MLX_AUDIO: tuple = (CueType.check_type("MLX_AUDIO"), "MLX_AUDIO", [])  # Blaizzy/mlx-audio
    MLX_LM: tuple = (has_api("MLX_LM"), "MLX_LM", [])  # "ml-explore/mlx-lm"
    MLX: tuple = (
        has_api("MFLUX") and has_api("MLX_LM") and CueType.check_type("MLX_AUDIO"),
        "MLX",
        [],
    )
    PARLER_TTS: tuple = (has_api("PARLER_TTS"), "PARLER_TTS", ["huggingface/parler-tts"])
    PLEIAS: tuple = (has_api("PLEIAS"), "PLEIAS", ["exdysa/Pleias-Pleias-RAG-Library"])  # bypasses vllm for macos to avoid requiring gcc/AVIX
    ORPHEUS_TTS: tuple = (has_api("ORPHEUS_TTS"), "ORPHEUS_TTS", ["canopyai/Orpheus-TTS"])
    OUTETTS: tuple = (has_api("OUTETTS"), "OUTETTS", ["edwko/OuteTTS"])
    SENTENCE_TRANSFORMERS: tuple = (has_api("SENTENCE_TRANSFORMERS"), "SENTENCE_TRANSFORMERS", [])  # UKPLab/sentence-transformers
    SHOW_O: tuple = (has_api("SHOW_O"), "SHOW_O", ["showlab/show-o"])
    SPANDREL: tuple = (has_api("SPANDREL"), "SPANDREL", [])
    SPANDREL_EXTRA_ARCHES: tuple = (has_api("SPANDREL_EXTRA_ARCHES"), "SPANDREL_EXTRA_ARCHES", [])
    SVDQUANT: tuple = (has_api("NUNCHAKU"), "NUNCHAKU", ["mit-han-lab/nunchaku"])
    TORCH: tuple = (has_api("TORCH"), "TORCH", [])  # Possible that torch is NOT needed (mlx_lm, or some other unforeseen future )
    TORCHAUDIO: tuple = (has_api("TORCHAUDIO"), "TORCHAUDIO", [])
    TORCHVISION: tuple = (has_api("TORCHVISION"), "TORCHVISION", [])
    TRANSFORMERS: tuple = (has_api("TRANSFORMERS"), "TRANSFORMERS", [])


class ChipType(Enum):
    """Device constants
    CUDA, MPS, XPU, MTIA [Supported PkgTypes]\n
    :param _show_all: ...
    :param _show_ready: ...
    : param _show_pkgs: ...
    """

    @classmethod
    def _show_all(cls) -> List:
        """Show all POSSIBLE processor types"""
        atypes = [atype for atype in ChipType.__dict__ if "_" not in atype]
        return atypes

    @classmethod
    def _show_ready(cls, api_name: Optional[str] = None) -> Union[List[str], bool]:
        """Show all READY devices.\n
        :param api_name: Boolean check for the specific API by name, defaults to None
        :return: `bool
        """
        atypes = cls._show_all()
        if api_name:
            return api_name.upper() in list(getattr(cls, x)[1] for x in atypes if getattr(cls, x)[0] is True)
        return [getattr(cls, x)[1] for x in atypes if getattr(cls, x)[0] is True]

    @classmethod
    def _show_pkgs(cls) -> Union[List[PkgType], str]:
        """Return compatible PkgTypes for all available chipsets\n
        If no chipsets are detected, returns onlyCPU compatibility options\n
        :return: `PkgType`s for the available processors including CPU
        """
        pkg_names = getattr(cls, "CPU")[-1]
        atypes = cls._show_ready()
        if atypes not in ["CPU", "XPU", "MTIA"]:
            pkg_names = getattr(cls, next(iter(atypes)))[-1] + pkg_names
        return pkg_names


chip_types = [
    (
        "CUDA",
        "cuda",
        [
            PkgType.BAGEL,
            PkgType.BITSANDBYTES,
            PkgType.EXLLAMAV2,
            PkgType.F_LITE,
            PkgType.LUMINA_MGPT,
            PkgType.ORPHEUS_TTS,
            PkgType.OUTETTS,
        ],
    ),
    (
        "MPS",
        "mps",
        [
            PkgType.MFLUX,
            PkgType.MLX_AUDIO,
            PkgType.MLX_LM,
            PkgType.MLX,
            PkgType.BAGEL,
        ],
    ),
    ("XPU", "xpu", []),
    ("MTIA", "mtia", []),
]
setattr(ChipType, "_device", first_available)
accelerator = ChipType._device(assign=True, init=True, clean=True)  # pylint:disable=no-member, protected-access
for name, key, pkg_type in chip_types:
    setattr(ChipType, name, (key in str(accelerator), name, pkg_type))
setattr(
    ChipType,
    "CPU",
    (
        True,
        "CPU",
        [
            PkgType.AUDIOGEN,
            PkgType.PARLER_TTS,
            PkgType.HIDIFFUSION,
            PkgType.SENTENCE_TRANSFORMERS,
            PkgType.DIFFUSERS,
            PkgType.TRANSFORMERS,
            PkgType.TORCH,
        ],
    ),
)


# class PipeType(Enum):
#     MFLUX: tuple = (ChipType._show_ready("mps"), PkgType.check_type("MFLUX"), {"mir_tag": "flux"})  # pylint:disable=protected-access
# MFLUX: tuple = ("MPS" in ChipType._show_ready("mps"), PkgType.MFLUX, {"mir_tag": "flux"})  # pylint:disable=protected-access


# Experimental way to abstract/declarify complex process names
class GenTypeC(BaseModel):
    """
    Generative inference types in ***C***-dimensional order\n
    ***Comprehensiveness***, sorted from 'most involved' to 'least involved'\n
    The terms define 'artistic' and ambiguous operations\n

    :param clone: Copying identity, voice, exact mirror
    :param sync: Tone, tempo, color, quality, genre, scale, mood
    :param translate: A range of comprehensible approximations\n
    """

    clone: Annotated[Callable | None, Field(default=None)]
    sync: Annotated[Callable | None, Field(default=None)]
    translate: Annotated[Callable | None, Field(default=None)]


class GenTypeCText(BaseModel):
    """
    Generative inference types in ***C***-dimensional order for text operations\n
    ***Comprehensiveness***, sorted from 'most involved' to 'least involved'\n
    The terms define 'concrete' and more rigid operations\n

    :param research: Quoting, paraphrasing, and deriving from sources
    :param chain_of_thought: A performance of processing step-by-step (similar to `reasoning`)
    :param question_answer: Basic, straightforward responses\n
    """

    research: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    chain_of_thought: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    question_answer: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]


class GenTypeE(BaseModel):
    """
    Generative inference operation types in ***E***-dimensional order \n
    ***Equivalence***, lists sorted from 'highly-similar' to 'loosely correlated.'"\n
    :param universal: Affecting all conversions
    :param text: Text-only conversions\n

    ***multimedia generation***
    ```
    Y-axis: Detail (Most involved to least involved)
    │
    │                             clone
    │                 sync
    │ translate
    │
    +───────────────────────────────────────> X-axis: Equivalence (Loosely correlated to highly similar)
    ```
    ***text generation***
    ```
    Y-axis: Detail (Most involved to least involved)
    │
    │                           research
    │             chain-of-thought
    │ question/answer
    │
    +───────────────────────────────────────> X-axis:  Equivalence (Loosely correlated to highly similar)
    ```

    This is essentially the translation operation of C types, and the mapping of them to E \n

    An abstract generalization of the set of all multimodal generative synthesis processes\n
    The sum of each coordinate pair reflects effective compute use\n
    In this way, both C types and their similarity are translatable, but not 1:1 identical\n
    Text is allowed to perform all 6 core operations. Other media perform only 3.\n
    """

    # note: `sync` may have better terms, such as 'harmonize' or 'attune'. `sync` was chosen because it is shorter

    universal: GenTypeC = GenTypeC(clone=None, sync=None, translate=None)
    text: GenTypeCText = GenTypeCText(research=None, chain_of_thought=None, question_answer=None)


# Here, a case could be made that tasks could be determined by filters, rather than graphing
# This is valid but, it offers no guarantees for difficult logic conditions that can be easily verified by graph algorithms
# Using graphs also allows us to offload the logic elsewhere

VALID_CONVERSIONS = ["text", "image", "music", "speech", "video", "3d_render", "vector_graphic", "upscale_image"]
VALID_JUNCTIONS = [""]

# note : decide on a way to keep paired tuples and sets together inside config dict
VALID_TASKS = {
    CueType.CORTEX: {
        ("text", "text"): ["text"],
    },
    CueType.VLLM: {
        ("text", "text"): ["text"],
        ("image", "text"): ["vision"],
    },
    CueType.OLLAMA: {
        ("text", "text"): ["mllama", "llava", "vllm"],
    },
    CueType.LLAMAFILE: {
        ("text", "text"): ["text"],
    },
    CueType.LM_STUDIO: {
        # ("image", "text"): [("vision", True)],
        ("text", "text"): ["llm"],
    },
    CueType.HUB: {
        ("text", "image"): ["Kolors", "image-generation"],
        ("image", "text"): ["image-generation", "image-text-to-text", "visual-question-answering"],
        ("text", "text"): ["chat", "conversational", "text-generation", "text2text-generation"],
        ("text", "video"): ["video generation"],
        ("speech", "text"): ["speech-translation", "speech-summarization", "automatic-speech-recognition", "dictation"],
        ("image", "video"): ["reference-to-video", "refernce-to-video"],
    },
    CueType.KAGGLE: {
        ("text", "text"): ["text"],
    },
}
