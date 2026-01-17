# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_import_module(mocker):
    """Fixture to mock import_module and simulate different module scenarios."""
    return mocker.patch("mir.config.conversion.import_submodules")


@pytest.fixture
def mock_pkgutil_iter_modules(mocker):
    """Fixture to mock pkgutil.iter_modules for controlled testing."""

    return mocker.patch(
        "pkgutil.iter_modules",
        return_value=[
            (Mock(), "allegro", True),
            (Mock(), "amused", True),
            (Mock(), "animatediff", True),
            (Mock(), "audioldm", True),
            (Mock(), "cogvideo", True),
            (Mock(), "deepfloyd_if", True),
        ],
    )


def test_list_diffusers_models():
    from mir.inspect.metadata import find_diffusers_docstrings

    find_diffusers_docstrings()


def test_find_docstrings_excluded(mock_import_module, mock_pkgutil_iter_modules):
    """Test that excluded modules are not processed."""
    from mir.inspect.metadata import find_diffusers_docstrings

    excluded_modules = ["ddpm"]

    def side_effect(import_name, *args, **kwargs):
        if any(exc in import_name for exc in excluded_modules):
            raise ImportError(f"Module {import_name} is excluded.")
        return Mock()

    mock_import_module.side_effect = side_effect
    results = list(find_diffusers_docstrings())  # type: ignore # noqa
    assert not any("ddpm" in call_arg[0][0] for call_arg in mock_import_module.call_args_list)
