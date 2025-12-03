#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0 */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import pytest
from mir.inspect.classes import resolve_code_names


def test_diffusers_name():
    assert resolve_code_names("StableDiffusionPipeline", "diffusers") == "stable-diffusion"


def test_transformers_name():
    assert resolve_code_names("BertModel", "transformers") == "bert"


def test_no_class():
    result = resolve_code_names()
    assert isinstance(result, list) is True
    assert len(result) > 300


def test_invalid_package():
    with pytest.raises(KeyError):
        assert resolve_code_names("EBertModel", "invalid_package") == ""


def test_mixed_search():
    assert resolve_code_names("EBertModel", "transformers") == ""


def test_difficult_search():
    assert resolve_code_names("AllegroPipeline", "diffusers") == "allegro"


def test_diff_folder_search():
    assert resolve_code_names("AllegroPipeline", "diffusers", path_format=True) == ["diffusers", "pipelines", "allegro"]


def test_tf_folder_search():
    assert resolve_code_names("Wav2Vec2Model", "transformers", path_format=True) == ["transformers", "models", "wav2vec2"]


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
