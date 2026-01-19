def test_info_key_exists_and_library_is_not_nested():
    from mir.generate.transformers.harvest import HarvestClasses

    Mir = HarvestClasses().db.db

    print(Mir.info.cnn.yolos)
    result = Mir.info.cnn.yolos["transformers"]  # should not throw
    assert result == {"repo": "hustvl/yolos-base", "model": "ops.cnn.yolos"}


def test_ops_key_exists_and_library_is_not_tested():
    from mir.generate.transformers.harvest import HarvestClasses

    Mir = HarvestClasses().db.db

    print(Mir.ops.cnn.yolos)
    result = Mir.ops.cnn.yolos["transformers"]  # should not throw
    assert result["model"] == "transformers.models.yolos.modeling_yolos.YolosModel"
    expected_tasks = [
        "YolosPreTrainedModel",
        "YolosForObjectDetection",
        "YolosImageProcessorFast",
        "YolosImageProcessor",
    ]
    assert all(task in result["tasks"] for task in expected_tasks)


def test_ops_tokenizer_created():
    from mir.generate.transformers.harvest import HarvestClasses

    Mir = HarvestClasses().db.db

    result = Mir.ops.encoder.tokenizer.zamba2["transformers"]
    assert result == {"model": "transformers.models.llama.tokenization_llama.LlamaTokenizer"}
