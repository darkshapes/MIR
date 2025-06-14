#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from collections import namedtuple
import unittest
import pytest
from unittest.mock import MagicMock, patch, Mock
import pytest_asyncio
from enum import Enum
from unittest.mock import patch


def has_api(name: str):
    # Mock implementation that always returns True
    return True


class CueType(Enum):
    """API library constants"""

    # Integers are used to differentiate boolean condition

    OLLAMA: tuple = (has_api("OLLAMA"), "OLLAMA")
    HUB: tuple = (has_api("HUB"), "HUB")
    LM_STUDIO: tuple = (has_api("LM_STUDIO"), "LM_STUDIO")
    CORTEX: tuple = (has_api("CORTEX"), "CORTEX")
    LLAMAFILE: tuple = (has_api("LLAMAFILE"), "LLAMAFILE")
    VLLM: tuple = (has_api("VLLM"), "VLLM")


@pytest_asyncio.fixture(loop_scope="session")
async def mock_has_api():
    with patch("mir.constants.has_api", return_value=True) as mocked:
        yield mocked


@pytest.mark.filterwarnings("ignore:open_text")
@pytest.mark.filterwarnings("ignore::DeprecationWarning:")
@patch("mir.constants.has_api", side_effect=lambda x: False)
def test_cuetype(mock_has_api):
    assert CueType.OLLAMA.value[0] is True
    assert CueType.HUB.value[0] is True
    assert CueType.LM_STUDIO.value[0] is True
    assert CueType.CORTEX.value[0] is True
    assert CueType.LLAMAFILE.value[0] is True
    assert CueType.VLLM.value[0] is True


@pytest_asyncio.fixture(loop_scope="session", name="mock_config")
def mock_deco():
    def decorator(func):
        def wrapper(*args, **kwargs):
            data = {
                "OLLAMA": {"api_kwargs": {"key1": "value1"}},
                "HUB": {"api_kwargs": {"key2": "value2"}},
                "LM_STUDIO": {"api_kwargs": {"key3": "value3"}},
                "CORTEX": {"api_kwargs": {"key4": "value4"}},
                "LLAMAFILE": {"api_kwargs": {"key5": "value5"}},
                "VLLM": {"api_kwargs": {"key6": "value6"}},
            }
            return data

        return wrapper

    return decorator


def cuetype_config_fixture():
    with patch("zodiac.chat_machine.CUETYPE_CONFIG", MagicMock()):
        yield mock_deco


@pytest.mark.asyncio(loop_scope="session")
async def test_lookup_cuetypes(mock_has_api):
    from mir.json_cache import JSONCache

    import os

    model = "🤡"

    for library in CueType.__members__.keys():
        library = getattr(CueType, library)
        # with patch("nnll_11.CueType", autocast=True):
        # req_form = await get_api(model, library)
        test_path = os.path.dirname(os.path.abspath(__file__))
        data = JSONCache(os.path.join(os.path.dirname(test_path), "mir", "config", "cuetype.json"))
        data._load_cache()
        expected = vars(data).get("_cache")
        assert isinstance(expected, dict)
        print(expected[library.value[1]])
        # assert expected == {"model": model, **expected[library.value[1]].get("api_kwargs")}
