### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->### <!-- // /*  d a r k s h a p e s */ -->


from enum import Enum
from typing import Annotated, Callable, List, Optional

from pydantic import BaseModel, Field

from nnll.monitor.file import dbug, nfo
from nnll.configure.init_gpu import first_available
from mir.json_cache import JSONCache, LIBTYPE_PATH_NAMED  # pylint:disable=no-name-in-module
from nnll.monitor.console import pretty_tabled_output

LIBTYPE_CONFIG = JSONCache(LIBTYPE_PATH_NAMED)

print("\n\n\n\n")


@LIBTYPE_CONFIG.decorator
def has_api(api_name: str, data: dict = None) -> bool:
    """Check available modules, try to import dynamically.
    True for successful import, else False

    :param api_name: Constant name for API
    :param _data: filled by config decorator, ignore, defaults to None
    :return: _description_
    """
    from json.decoder import JSONDecodeError

    def set_false(api_name):
        pretty_tabled_output(
            "api",
            {"unavailable": api_name, "": ""},
        )
        return False

    def set_true(api_name):
        pretty_tabled_output("api", {"found": api_name, "": ""})
        return True

    def check_host(api_name) -> bool:
        import httpcore
        import httpx
        from urllib3.exceptions import NewConnectionError, MaxRetryError
        import requests

        dbug(f"Response from API  {api_data}")
        try:
            if api_data.get("api_url", 0):
                request = requests.get(api_data.get("api_url"), timeout=(1, 1))
                if request is not None:
                    if hasattr(request, "reason") and request.reason == "OK":  # The curious case of Ollama
                        set_true(api_name)
                    status = request.json()
                    if status.get("result") == "OK":
                        nfo(f"Found {api_name}")
                        set_true(api_name)
                set_false(api_name)
        except JSONDecodeError as error_log:
            dbug(error_log)
            dbug(f"json for ! {api_data}")
            dbug(request.status_code)
            if request.ok:
                nfo(f"Found {api_name}")
                set_true(api_name)
            try:
                request.raise_for_status()
            except requests.HTTPError() as _error_log:
                dbug(_error_log)
                set_false(api_name)

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
        ):
            set_false(api_name)
        set_false(api_name)

    try:
        api_data = data.get(api_name, False)  # pylint: disable=unsubscriptable-object
    except JSONDecodeError as error_log:
        dbug(error_log)
    if not api_data:
        api_data = {"module": api_name.lower()}

    if api_name == "LM_STUDIO":
        try:
            from lmstudio import APIConnectionError, APITimeoutError, APIStatusError, LMStudioWebsocketError

            try:
                return check_host(api_name)
            except (LMStudioWebsocketError, APIConnectionError, APITimeoutError, APIStatusError, JSONDecodeError):
                return set_false(api_name)
        except (UnboundLocalError, ImportError, ModuleNotFoundError, JSONDecodeError):
            return set_false(api_name)
    elif api_name in ["LLAMAFILE", "CORTEX"]:
        try:
            from openai import APIConnectionError, APIStatusError, APITimeoutError

            try:
                return check_host(api_name)
            except (APIConnectionError, APITimeoutError, APIStatusError, JSONDecodeError):
                return set_false(api_name)
        except (UnboundLocalError, ImportError, ModuleNotFoundError, JSONDecodeError):
            return set_false(api_name)
    elif api_name in ["OLLAMA"]:
        try:
            return check_host(api_name)
        except (UnboundLocalError, ImportError, ModuleNotFoundError, JSONDecodeError):
            return set_false(api_name)
    else:
        try:
            __import__(api_data.get("module"))
            if api_name not in ["OLLAMA", "LM_STUDIO", "CORTEX", "LLAMAFILE", "VLLM"]:
                set_true(api_name)
        except (UnboundLocalError, ImportError, ModuleNotFoundError):
            set_false(api_name)
        return check_host(api_name)


class BaseEnum(Enum):
    """Base class for available system packages\n"""

    @classmethod
    def show_all(cls) -> List:
        """Show all possible API types"""
        return [x for x, y in LibType.__members__.items()]

    @classmethod
    def show_available(cls) -> bool:
        """Show all available API types"""
        return [library.value[1] for library in list(cls) if library.value[0] is True]

    @classmethod
    def check_type(cls, type_name: str, server: bool = False) -> bool:
        """Check for a single API"""
        type_name = type_name.upper()
        if hasattr(cls, type_name):
            available = next(iter(getattr(cls, type_name).value))
            if server and available:
                return has_api(type_name)
            return available


class LibType(BaseEnum):
    """API library constants
    <NAME: (Availability, IMPORT_NAME)>"""

    # Integers are usedto differentiate boolean condition
    # GIVEN : The state of all library modules & servers are marked at launch

    OLLAMA: tuple = (has_api("OLLAMA"), "OLLAMA")
    HUB: tuple = (has_api("HUB"), "HUB")
    LM_STUDIO: tuple = (has_api("LM_STUDIO"), "LM_STUDIO")
    CORTEX: tuple = (has_api("CORTEX"), "CORTEX")
    LLAMAFILE: tuple = (has_api("LLAMAFILE"), "LLAMAFILE")
    VLLM: tuple = (has_api("VLLM"), "VLLM")
    MLX_AUDIO: tuple = (has_api("MLX_AUDIO"), "MLX_AUDIO")


example_str = ("function_name", "import.function_name")


class PkgType(BaseEnum):
    """Package dependency constants
    <NAME: (Availability, IMPORT_NAME)>"""

    AUDIOGEN: tuple = (has_api("AUDIOCRAFT"), "AUDIOCRAFT")
    BAGEL: tuple = (has_api("BAGEL"), "BAGEL")
    BITSANDBYTES: tuple = (has_api("BITSANDBYTES"), "BITSANDBYTES")
    DIFFUSERS: tuple = (has_api("DIFFUSERS"), "DIFFUSERS")
    F_LITE: tuple = (has_api("F_LITE"), "F_LITE")
    HIDIFFUSION: tuple = (has_api("HIDIFFUSION"), "HIDIFFUSION")
    LUMINA_MGPT: tuple = (has_api("INFERENCE_SOLVER"), "INFERENCE_SOLVER")  # Alpha vllm
    MFLUX: tuple = (has_api("MFLUX"), "MFLUX")  # pylint:disable=no-member
    MLX_AUDIO: tuple = LibType.MLX_AUDIO.value
    ORPHEUS_TTS: tuple = (has_api("ORPHEUS_TTS"), "ORPHEUS_TTS")
    OUTETTS: tuple = (has_api("OUTETTS"), "OUTETTS")
    SENTENCE_TRANSFORMERS: tuple = (has_api("SENTENCE_TRANSFORMERS"), "SENTENCE_TRANSFORMERS")
    TORCH: tuple = (has_api("TORCH"), "TORCH")
    TORCHAUDIO: tuple = (has_api("TORCHAUDIO"), "TORCHAUDIO")
    TORCHVISION: tuple = (has_api("TORCHVISION"), "TORCHVISION")
    TRANSFORMERS: tuple = (has_api("TRANSFORMERS"), "TRANSFORMERS")


class ChipType(Enum):
    """Device constants
    CUDA, MPS, XPU, MTIA
    """

    @classmethod
    def _show_all(cls) -> List:
        atypes = [atype for atype in ChipType.__dict__ if "_" not in atype]
        return atypes

    @classmethod
    def _show_ready(cls, api_name: Optional[str] = None):
        atypes = cls._show_all()
        if api_name:
            return api_name.upper() in list(getattr(cls, x)[1] for x in atypes if getattr(cls, x)[0] is True)
        return [getattr(cls, x)[1] for x in atypes if getattr(cls, x)[0] is True]


chip_types = [
    ("CUDA", "cuda"),
    ("MPS", "mps"),
    ("XPU", "xpu"),
    ("MTIA", "mtia"),
]
accelerator = first_available(assign=False)

for name, key in chip_types:
    setattr(ChipType, name, (key in accelerator, name))
setattr(ChipType, "CPU", (True, "CPU"))


class PipeType(Enum):
    MFLUX: tuple = (ChipType._show_ready("mps"), PkgType.check_type("MFLUX"), {"mir_tag": "flux"})  # pylint:disable=protected-access
    # MFLUX: tuple = ("MPS" in ChipType._show_ready("mps"), PkgType.MFLUX, {"mir_tag": "flux"})  # pylint:disable=protected-access


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
    LibType.CORTEX: {
        ("text", "text"): ["text"],
    },
    LibType.VLLM: {
        ("text", "text"): ["text"],
        ("image", "text"): ["vision"],
    },
    LibType.OLLAMA: {
        ("text", "text"): ["mllama", "llava", "vllm"],
    },
    LibType.LLAMAFILE: {
        ("text", "text"): ["text"],
    },
    LibType.LM_STUDIO: {
        # ("image", "text"): [("vision", True)],
        ("text", "text"): ["llm"],
    },
    LibType.HUB: {
        ("text", "image"): ["Kolors", "image-generation"],
        ("image", "text"): ["image-generation", "image-text-to-text", "visual-question-answering"],
        ("text", "text"): ["chat", "conversational", "text-generation", "text2text-generation"],
        ("text", "video"): ["video generation"],
        ("speech", "text"): ["speech-translation", "speech-summarization", "automatic-speech-recognition"],
        ("image", "video"): ["reference-to-video", "refernce-to-video"],  # typos: ignore
    },
}
