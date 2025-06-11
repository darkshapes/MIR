import pytest
from unittest.mock import patch, MagicMock
from mir.generators import create_pipe_entry, mir_label  # Replace 'your_module' with the actual module name


# Mocking the root_class and mir_label functions since they are not provided
def root_class(pipe_data):
    """Mock function to simulate sub_classes retrieval."""
    if "unet" in pipe_data:
        return ["unet", "encoder"]
    elif "transformer" in pipe_data:
        return ["transformer", "decoder"]
    else:
        return []


@pytest.fixture
def mock_diffusers():
    with patch("diffusers", autocast=True) as mocked:
        return mocked


# Test case 1: Basic test with a standard pipeline class and no special keywords
def test_create_pipe_entry_standard_case_prior_underscore():
    repo_path = "standard_org/standard_repo-prior"
    pipe_class = "StableDiffusionPipeline"

    result = create_pipe_entry(repo_path, pipe_class)

    assert len(result) == 2
    mir_series, prefixed_data = result

    assert mir_series == "info.unet.standard-repo"
    assert "repo" in prefixed_data.get("prior")
    assert prefixed_data["prior"]["repo"] == repo_path
    assert "pkg" in prefixed_data.get("prior")
    assert prefixed_data["prior"]["pkg"][0]["diffusers"] == pipe_class


# Test case 2: Test with a pipeline class that includes 'unet' in sub_classes
def test_create_pipe_entry_unet_case():
    # from unittest.mock import patch, MagicMock
    # import diffusers

    repo_path = "standard_org/default_series"
    pipe_class = "ConsisIDPipeline"

    result = create_pipe_entry(repo_path, pipe_class)

    assert len(result) == 2
    mir_series, prefixed_data = result

    assert mir_series == "info.dit.default-series"
    assert prefixed_data["base"]["repo"] == repo_path
    assert prefixed_data["base"]["pkg"][0]["diffusers"] == pipe_class


def test_create_pipe_entry_transformer_case():
    repo_path = "default_series/default_repo"
    pipe_class = "HunyuanDiTPipeline"

    result = create_pipe_entry(repo_path, pipe_class)

    mir_series, prefixed_data = result

    assert mir_series == "info.dit.default-repo"
    assert prefixed_data["base"]["repo"] == repo_path
    assert prefixed_data["base"]["pkg"][0]["diffusers"] == pipe_class


# # Test case 4: Test with a pipeline class that includes 'kandinsky' in the repo path
def test_create_pipe_entry_kandinsky_case():
    repo_path = "kandinsky_series/kandinsky_repo-v1"
    pipe_class = "KandinskyPipeline"

    result = create_pipe_entry(repo_path, pipe_class)

    mir_series, prefixed_data = result

    assert mir_series == "info.unet.kandinsky-repo"
    assert prefixed_data["v1"]["repo"] == repo_path
    assert prefixed_data["v1"]["pkg"][0]["diffusers"] == pipe_class


def test_create_pipe_entry_shap_e_case():
    repo_path = "openai/shap-e_40496"
    pipe_class = "ShapEPipeline"

    result = create_pipe_entry(repo_path, pipe_class)

    mir_series, prefixed_data = result

    assert mir_series == "info.unet.shap-e"
    assert prefixed_data["40496"]["repo"] == repo_path
    assert prefixed_data["40496"]["pkg"][0]["diffusers"] == pipe_class


# Test case 6: Test with a pipeline class that is 'FluxPipeline'
def test_create_pipe_entry_flux_case():
    repo_path = "cocktailpeanut/xulf-schnell"
    pipe_class = "FluxPipeline"

    result = create_pipe_entry(repo_path, pipe_class)

    mir_series, prefixed_data = result

    assert mir_series == "info.dit.xulf"
    assert prefixed_data["schnell"]["repo"] == repo_path
    assert prefixed_data["schnell"]["pkg"][0]["diffusers"] == pipe_class
    assert 1 in prefixed_data["schnell"]["pkg"]
    assert prefixed_data["schnell"]["pkg"][1]["mflux"] == "Flux1"


# # Test case 7: Edge case with an empty repo path and a non-standard pipeline class
def test_create_pipe_entry_edge_case():
    repo_path = ""
    pipe_class = "StableDiffusionPipeline"
    with pytest.raises(TypeError) as exc_info:
        result = create_pipe_entry(repo_path, pipe_class)

    assert isinstance(exc_info.value, TypeError)
    assert exc_info.value.args == ("'repo_path'  or 'pipe_class' StableDiffusionPipeline unset",)


# # Test case 8: Test with a pipeline class that includes 'prior' in sub_classes
def test_create_pipe_entry_prior_case():
    repo_path = "babalityai/finish_him_.kascade_prior"
    pipe_class = "StableCascadePriorPipeline"

    result = create_pipe_entry(repo_path, pipe_class)

    mir_series, prefixed_data = result

    assert mir_series == "info.unet.finish-him--kascade"
    assert prefixed_data["prior"]["repo"] == repo_path
    assert prefixed_data["prior"]["pkg"][0]["diffusers"] == pipe_class


# # Test case 9: Test with a pipeline class that includes 'decoder' in sub_classes
def test_create_pipe_entry_decoder_case():
    repo_path = "babalityai/finish_him_.kascade"
    pipe_class = "StableCascadeDecoderPipeline"

    result = create_pipe_entry(repo_path, pipe_class)

    mir_series, prefixed_data = result

    assert mir_series == "info.unet.finish-him--kascade"
    assert prefixed_data["base"]["repo"] == repo_path
    assert prefixed_data["base"]["pkg"][0]["diffusers"] == pipe_class


# # Test case 11: Test with a pipeline class that includes multiple sub_classes (e.g., 'unet' and 'transformer')
# def test_create_pipe_entry_multiple_subclasses_case():
#     # This is a hypothetical scenario since the root_class function would return only one type

#     repo_path = "pretenderz/pretend_repo"
#     pipe_class = "DiffusionPipeline"
#     mock_unet_pipeline = MagicMock()

#     class FakeAttrib:
#         transformer = "shuffled"
#         unet = "also_shuffled"

#     mock_unet_pipeline.side_effect = FakeAttrib
#     mock_unet_pipeline.return_value = FakeAttrib

#     with patch.dict("sys.modules", {"diffusers": MagicMock()}):
#         with patch("diffusers.pipelines.pipeline_utils.DiffusionPipeline") as mock_unet_pipeline:
#             result = create_pipe_entry(repo_path, pipe_class)

#     result = create_pipe_entry(repo_path, pipe_class)

#     mir_series, prefixed_data = result
#     # print(result)
#     assert mir_series == "info.unet.pretend_repo"  # Since 'unet' takes precedence in the logic
#     assert prefixed_data["repo"] == repo_path
#     assert prefixed_data["pkg"][0]["diffusers"] == pipe_class
