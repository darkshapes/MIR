import unittest
from mir.doc_parser import DocParser, parse_docs

CONSISTENCY_TEST_DOC = """
    Examples:
        ```py
        >>> import torch

        >>> from diffusers import ConsistencyModelPipeline

        >>> device = "cuda"
        >>> # Load the cd_imagenet64_l2 checkpoint.
        >>> model_id_or_path = "openai/diffusers-cd_imagenet64_l2"
        >>> pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        >>> pipe.to(device)

        >>> # Onestep Sampling
        >>> image = pipe(num_inference_steps=1).images[0]
        >>> image.save("cd_imagenet64_l2_onestep_sample.png")

        >>> # Onestep sampling, class-conditional image generation
        >>> # ImageNet-64 class label 145 corresponds to king penguins
        >>> image = pipe(num_inference_steps=1, class_labels=145).images[0]
        >>> image.save("cd_imagenet64_l2_onestep_sample_penguin.png")

        >>> # Multistep sampling, class-conditional image generation
        >>> # Timesteps can be explicitly specified; the particular timesteps below are from the original GitHub repo:
        >>> # https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L77
        >>> image = pipe(num_inference_steps=None, timesteps=[22, 0], class_labels=145).images[0]
        >>> image.save("cd_imagenet64_l2_multistep_sample_penguin.png")
        ```
"""

CASCADE_TEST_DOC = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline

        >>> prior_pipe = StableCascadePriorPipeline.from_pretrained(
        ...     "stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16
        ... ).to("cuda")
        >>> gen_pipe = StableCascadeDecoderPipeline.from_pretrain(
        ...     "stabilityai/stable-cascade", torch_dtype=torch.float16
        ... ).to("cuda")

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> prior_output = pipe(prompt)
        >>> images = gen_pipe(prior_output.image_embeddings, prompt=prompt)
        ```
"""


class TestDocParser(unittest.TestCase):
    def test_parse_simple_case(self):
        doc_string = """
            >>> pipe = MyPipeline.from_pretrained("model/repo")
        """
        result = parse_docs(doc_string)
        self.assertEqual(result[0], "MyPipeline")  # pipe_class
        self.assertEqual(result[1], "model/repo")  # repo_path
        self.assertIsNone(result[2])  # staged_class
        self.assertIsNone(result[3])  # staged_repo

    def test_parse_with_variable_resolution(self):
        doc_string = """
            model_id = "custom/model"
            >>> pipe = MyPipeline.from_pretrained(model_id)
        """
        result = parse_docs(doc_string)
        self.assertEqual(result[0], "MyPipeline")
        self.assertEqual(result[1], "custom/model")

    def test_parse_staged_case(self):
        doc_string = """
            >>> pipe = MyPipeline.from_pretrained("model/repo")
            >>> prior_pipe = PriorPipeline.from_pretrain("prior/repo")
        """
        result = parse_docs(doc_string)
        print(result)
        self.assertEqual(result[0], "MyPipeline")  # pipe_class
        self.assertEqual(result[1], "model/repo")  # repo_path
        self.assertEqual(result[2], "PriorPipeline")  # staged_class
        self.assertEqual(result[3], "prior/repo")  # staged_repo

    def test_parse_no_match(self):
        doc_string = """
            >>> something_else = SomeClass.do_something()
        """
        result = parse_docs(doc_string)
        self.assertIsNone(result[0])  # pipe_class
        self.assertIsNone(result[1])  # repo_path
        self.assertIsNone(result[2])  # staged_class
        self.assertIsNone(result[3])  # staged_repo

    def test_parse_multiline_doc(self):
        doc_string = """
            # model_id_or_path = "another/repo"
            >>> pipe_prior = PriorPipeline.from_pretrain(model_id_or_path)
            >>> pipeline = MyPipeline.from_pretrained("repo/path")
        """
        result = parse_docs(doc_string)
        self.assertEqual(result[0], "MyPipeline")  # pipe_class
        self.assertEqual(result[1], "repo/path")  # repo_path
        self.assertEqual(result[2], "PriorPipeline")  # staged_class
        self.assertEqual(result[3], "another/repo")  # staged_repo


if __name__ == "__main__":
    unittest.main()
