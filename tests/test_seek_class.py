# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from mir.config.conversion import import_submodules
from mir.inspect.pipes import get_class_parent_folder


def test_seek_diffusers_path():
    assert get_class_parent_folder(import_submodules("AllegroPipeline", "diffusers"), "diffusers") == ["diffusers", "pipelines", "allegro"]


def test_seek_transformers_path():
    assert get_class_parent_folder(import_submodules("AlbertModel", "transformers"), "transformers") == ["transformers", "models", "albert"]


def test_seek_class_attention():
    assert get_class_parent_folder("CogVideoXAttnProcessor2_0", "diffusers") is None
