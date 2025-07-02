import unittest
from mir.doc_parser import DocParser, parse_docs


class TestDocParser(unittest.TestCase):
    def test_parse_simple_case(self):
        doc_string = """
            >>> pipe = MyPipeline.from_pretrained("model/repo")
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "MyPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "model/repo")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_with_variable_resolution(self):
        doc_string = """
            model_id = "custom/model"
            >>> pipe = MyPipeline.from_pretrained(model_id)
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "MyPipeline")
        self.assertEqual(result.pipe_repo, "custom/model")

    def test_parse_staged_case(self):
        doc_string = """
            >>> pipe = MyPipeline.from_pretrained("model/repo")
            >>> prior_pipe = PriorPipeline.from_pretrain("prior/repo")
        """
        result = parse_docs(doc_string)
        print(result)
        self.assertEqual(result.pipe_class, "MyPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "model/repo")  # repo_path
        self.assertEqual(result.staged_class, "PriorPipeline")  # staged_class
        self.assertEqual(result.staged_repo, "prior/repo")  # staged_repo

    def test_parse_no_match(self):
        doc_string = """
            >>> something_else = SomeClass.do_something()
        """
        result = parse_docs(doc_string)
        self.assertIsNone(result)  # pipe_class

    def test_parse_multiline_doc(self):
        doc_string = """
            # model_id_or_path = "another/repo"
            >>> pipe_prior = PriorPipeline.from_pretrain(model_id_or_path)
            >>> pipeline = MyPipeline.from_pretrained("repo/path")
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "MyPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "repo/path")  # repo_path
        self.assertEqual(result.staged_class, "PriorPipeline")  # staged_class
        self.assertEqual(result.staged_repo, "another/repo")  # staged_repo

    def test_parse_blip(self):
        doc_string = """
        Examples:
            ```py
            >>> from diffusers.pipelines import BlipDiffusionPipeline
            >>> from diffusers.utils import load_image
            >>> import torch

            >>> blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
            ...     "Salesforce/blipdiffusion", torch_dtype=torch.float16
            ... ).to("cuda")


            >>> cond_subject = "dog"
            >>> tgt_subject = "dog"
            >>> text_prompt_input = "swimming underwater"

            >>> cond_image = load_image(
            ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/dog.jpg"
            ... )
            >>> guidance_scale = 7.5
            >>> num_inference_steps = 25
            >>> negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


            >>> output = blip_diffusion_pipe(
            ...     text_prompt_input,
            ...     cond_image,
            ...     cond_subject,
            ...     tgt_subject,
            ...     guidance_scale=guidance_scale,
            ...     num_inference_steps=num_inference_steps,
            ...     neg_prompt=negative_prompt,
            ...     height=512,
            ...     width=512,
            ... ).images
            >>> output[0].save("image.png")
            ```
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "BlipDiffusionPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "Salesforce/blipdiffusion")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo


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

PIXART_TEST_DOC = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import PixArtSigmaPipeline

        >>> # You can replace the checkpoint id with "PixArt-alpha/PixArt-Sigma-XL-2-512-MS" too.
        >>> pipe = PixArtSigmaPipeline.from_pretrained(
        ...     "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
        ... )
        >>> # Enable memory optimizations.
        >>> # pipe.enable_model_cpu_offload()

        >>> prompt = "A small cactus with a happy face in the Sahara desert."
        >>> image = pipe(prompt).images[0]
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

if __name__ == "__main__":
    unittest.main()
