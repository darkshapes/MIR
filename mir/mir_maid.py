### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""神经网络的数据注册"""

# pylint: disable=possibly-used-before-assignment, line-too-long
from typing import Any, Callable, List, Optional
import os

# from mir.constants import LibType
from nnll.monitor.file import debug_monitor, dbug, nfo
from mir.json_cache import JSONCache, MIR_PATH_NAMED  # pylint:disable=no-name-in-module
from mir.mir import mir_entry
# from pkg_detect import root_class


class MIRDatabase:
    """Machine Intelligence Resource Database"""

    database: Optional[dict[str, Any]]
    mir_file = JSONCache(MIR_PATH_NAMED)

    def __init__(self) -> None:
        self.read_from_disk()

    @debug_monitor
    def add(self, resource: dict[str, Any]) -> None:
        """Merge pre-existing MIR entries, or add new ones
        :param element: _description_
        """
        parent_key = next(iter(resource))
        if self.database is not None:
            if self.database.get(parent_key, 0):
                self.database[parent_key] = {**self.database[parent_key], **resource[parent_key]}
            else:
                self.database[parent_key] = resource[parent_key]

    @mir_file.decorator
    def write_to_disk(self, data: Optional[dict] = None) -> None:  # pylint:disable=unused-argument
        """Save data to JSON file\n"""
        # from pprint import pprint
        self.mir_file.update_cache({"..": ".."}, replace=True)
        self.mir_file.update_cache(self.database, replace=True)
        self.database = self.read_from_disk()
        # nfo(self.database)
        nfo(f"Wrote {len(self.database)} lines to MIR database file.")

    @mir_file.decorator
    def read_from_disk(self, data: Optional[dict] = None) -> dict[str, Any]:
        """Populate mir database\n
        :param data: mir decorater auto-populated, defaults to None
        :return: dict of MIR data"""
        self.database = data
        return self.database

    @debug_monitor
    def _ready_value(self, value: str, target: str, series: str, compatibility: str) -> List[str]:
        """Process a single value for matching against the target\n
        :param value: An unknown string value
        :param target: The search target
        :param series: MIR URI domain.arch.series identifier
        :param compatibility: MIR URI compatibility identifier
        :return: _description_
        """
        results = []
        if isinstance(value, str):
            value = [value]
        for option in value:
            option_lower = option.lower()
            if option_lower == target:
                return [option, series, compatibility, True]
            elif target in option_lower:
                results.append([option, series, compatibility, False])
        return results

    @staticmethod
    def grade_char_match(matches: List[List[str]], target: str) -> list[str, str]:
        """Evaluate and select the best match from a list of potential matches\n
        :param matches: Possible matches to compare
        :param target: Desired entry to match
        :return: The closest matching dictionary elements
        """
        if not matches:
            return None
        min_gap = float("inf")
        best_match = None
        for match in matches:
            option, series, compatibility, _ = match
            if target in option or option in target:
                max_len = len(os.path.commonprefix([option, target]))
                gap = abs(len(option) - len(target)) + (len(option) - max_len)
                if gap < min_gap:
                    min_gap = gap
                    best_match = [series, compatibility]
        return best_match

    @debug_monitor
    def find_path(self, field: str, target: str) -> list[str]:
        """Retrieve MIR path based on nested value search\n
        :param field: Known field to look within
        :param target: Search pattern for field
        :param join_tag: Combine tag elements, defaults to False
        :return: A list or string of the found tag
        :raises KeyError: Target string not found
        """
        target = target.lower()
        matches = []
        for series, comp in self.database.items():
            for compatibility, fields in comp.items():
                value = fields.get(field)
                if value is not None:
                    match_results = self._ready_value(value, target, series, compatibility)
                    if next(iter(match_results), 0):
                        if next(iter(match_results))[3]:
                            best_match = [series, compatibility]
                            return best_match
                        matches.extend(match_results)
        best_match = self.grade_char_match(matches, target)
        if best_match:
            dbug(best_match)
            return best_match
        raise KeyError(f"Query '{target}' not found when searched {len(self.database)}'{field}' options")


def build_mir_unet(mir_db: MIRDatabase):
    """Create mir unet info database"""

    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="base",
            gen_kwargs={
                "diffusers": {
                    "num_inference_steps": 40,
                    "denoising_end": 0.8,
                    "output_type": "latent",
                }
            },
            init_kwargs={
                "diffusers": {
                    "use_safetensors": True,
                }
            },
            layer_256=["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
            weight_map="weight_maps/model.unet.stable-diffusion-xl:base.json",
            repo=["stabilityai/stable-diffusion-xl-base-1.0"],
            alt_pipe={"diffusers": ["DiffusionPipeline"]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="refiner",
            repo=["stabilityai/stable-diffusion-xl-refiner-1.0"],
            layer_256=["8c2d0d32cff5a74786480bbaa932ee504bb140f97efdd1a3815f14a610cf6e4a"],
            weight_map="weight_maps/stable-diffusion-xl-refiner.json",
            package={"diffusers": "DiffusionPipeline", "num_inference_steps": 40, "denoising_end": 0.8},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="kolors",
            comp="diffusers",
            repo=["kwai-kolors/kolors-diffusers"],
            fits=["ops.precision.f16"],
            package={"diffusers": "KolorsPipeline", "negative_prompt": "", "guidance_scale": 5.0, "num_inference_steps": 50},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-cascade",
            comp="combined",
            repo=["stabilityai/stable-cascade"],
            package={"diffusers": ["StableCascadeCombinedPipeline"]},
            gen_kwargs={"negative_prompt": "", "num_inference_steps": 10, "prior_num_inference_steps": 20, "prior_guidance_scale": 3.0, "width": 1024, "height": 1024},
            init_kwargs={"variant": "bf16", "torch_dtype": "torch.bfloat16"},
        )
    )

    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-cascade",
            comp="prior",
            repo=["stabilityai/stable-cascade-prior"],
            dep_alt={"diffusers": ["StableCascadePriorPipeline"]},
            layer_256=[
                "2b6986954d9d2b0c702911504f78f5021843bd7050bb10444d70fa915cb495ea",
                "2aa5a461c4cd0e2079e81554081854a2fa01f9b876d7124c8fff9bf1308b9df7",
                "ce474fd5da12f1d465a9d236d61ea7e98458c1b9d58d35bb8412b2acb9594f08",
                "1b035ba92da6bec0a9542219d12376c0164f214f222955024c884e1ab08ec611",
                "22a49dc9d213d5caf712fbf755f30328bc2f4cbdc322bcef26dfcee82f02f147",
            ],
            init_kwargs={"variant": "bf16", "torch_dtype": "torch.bfloat16"},
            gen_kwargs={"height": 1024, "width": 1024, "negative_prompt": "", "guidance_scale": 4.0, "num_images_per_prompt": 1, "num_inference_steps": 20},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-cascade",
            comp="decoder",
            repo=["stabilityai/stable-cascade"],
            dep_apt={"diffusers": ["StableCascadeDecoderPipeline"]},
            layer_256=[
                "fde5a91a908e8cb969f97bcd20e852fb028cc039a19633b0e1559ae41edeb16f",
                "24fa8b55d12bf904878b7f2cda47c04c1a92da702fe149e28341686c080dfd4f",
                "a7c96afb54e60386b7d077bf3f00d04596f4b877d58e6a577f0e1a08dc4a0190",
                "f1300b9ffe051640555bfeee245813e440076ef90b669332a7f9fb35fffb93e8",
                "047fa405c9cd5ad054d8f8c8baa2294fbc663e4121828b22cb190f7057842a64",
            ],
            init_kwargs={"variant": "bf16", "torch_dtype": "torch.bfloat16"},
            gen_kwargs={"guidance_scale": 0.0, "output_type": "pil", "num_inference_steps": 10},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="pony-diffusion",
            layer_256=["d4fc7682a4ea9f2dfa0133fafb068f03fdb479158a58260dcaa24dcf33608c16"],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="animagine",
            layer_256=["31164c11db41b007f15c94651a8b1fa4d24097c18782d20fabe13c59ee07aa3a"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="illustrious",
            layer_256=["c4a8d365e7fe07c6dbdd52be922aa6dc23215142342e3e7f8f967f1a123a6982"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="turbo",
            layer_256=["fc94481f0c52b21c5ac1fdade8d9c5b210f7239253f86ef21e6198fe393ed60e"],
            file_256=["a599c42a9f4f7494c7f410dbc0fd432cf0242720509e9d52fa41aac7a88d1b69"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="playground-2.5-base",
            layer_256=["a6f31493ceeb51c88c5239188b9078dc64ba66d3fc5958ad48c119115b06120c"],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="playground-2.5-aesthetic",
            repo=["playgroundai/playground-v2.5-1024px-aesthetic"],
            layer_256=[
                "fe2e9edf7e3923a80e64c2552139d8bae926cc3b028ca4773573a6ba60e67c20",
                "d4813e9f984aa76cb4ac9bf0972d55442923292d276e97e95cb2f49a57227843",
            ],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
            init_kwargs={"torch_dtype": "torch.float16", "variant": "fp16"},
            gen_kwargs={"num_inference_steps": 50, "guidance_scale": 3},
        )
    )


def build_mir_dit(mir_db: MIRDatabase):
    """Create mir diffusion transformer info database"""

    # from nnll.configure.init_gpu import first_available

    # if "mps" in first_available(assign=False):
    #     try:
    #         from mflux import _import_structure

    #         for class_name in _import_structure["schedulers"]:
    #             for minor in ["Discrete", "Scheduler", "Multistep", "Solver"]:
    #                 series_name = class_name.replace(minor, "")
    #             series_name.lower()
    #             mir_db.add(
    #                 mir_entry(
    #                     domain="ops",
    #                     arch="scheduler",
    #                     series=series_name,
    #                     comp="[init]",
    #                     requires={"mflux": {class_name}},
    #                 )
    #             )
    #     except (ImportError, ModuleNotFoundError) as error_log:
    #         dbug(error_log)

    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="medium",
            repo=["stabilityai/stable-diffusion-3.5-medium", "adamo1139/stable-diffusion-3.5-medium-ungated"],
            layer_256=["dee29a467c44cff413fcf1c2dda0b31f5f0a4e093029a8e5a05140f40ae061ee"],
            dep_pkg={"diffusers": ["StableDiffusion3Pipeline"]},
            gen_kwargs={"num_inference_steps": 40, "guidance_scale": 4.5},
            init_kwargs={"torch_dtype": "torch.float16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="large",
            repo=["stabilityai/stable-diffusion-3.5-large", "adamo1139/stable-diffusion-3.5-large-ungated"],
            layer_256=["8c2e5bc99bc89290254142469411db66cb2ca2b89b129cd2f6982b30e26bd465"],
            gen_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5},
            init_kwargs={"torch_dtype": "torch.float16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="gguf",
            layer_256=["e7eddc3cd09ccf7c9c03ceef70bbcd91d44d46673857d37c3abfe4e6ee240a96"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="alchemist-large",
            repo=["yandex/stable-diffusion-3.5-large-alchemist"],
            gen_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="alchemist-medium",
            repo=["yandex/stable-diffusion-3.5-medium-alchemist"],
            gen_kwargs={"num_inference_steps": 40, "guidance_scale": 4.5},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="medium-turbo",
            repo=["tensorart/stable-diffusion-3.5-medium-turbo"],
            gen_kwargs={"num_inference_steps": 8, "guidance_scale": 1.0},
            scheduler=["ops.scheduler.flow-match"],
            scheduler_kwargs={"shift": 5},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="large-turbox",
            repo=["tensorart/stable-diffusion-3.5-large-TurboX"],
            gen_kwargs={"num_inference_steps": 8, "guidance_scale": 1.0},
            scheduler=["ops.scheduler.flow-match"],
            scheduler_kwargs={"shift": 5},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="schnell",
            repo=["black-forest-labs/flux.1-schnell", "cocktailpeanut/xulf-s"],
            layer_256=["ef5c9cd1ebe6e3be5e8b1347eca0a6f0b138986c71220a7f1c2c14f29d01beed"],
            dep_pkg={"diffusers": ["FluxPipeline"]},
            gen_kwargs={"guidance_scale": 0.0, "num_inference_steps": 4, "max_sequence_length": 256},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="shuttle-3.1-aesthetic",
            repo=["shuttleai/shuttle-3.1-aesthetic"],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 3.5, "num_inference_steps": 4, "max_sequence_length": 256},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="shuttle-3-diffusion",
            repo=["shuttleai/shuttle-3-diffusion"],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 3.5, "num_inference_steps": 4, "max_sequence_length": 256},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="dev",
            repo=["black-forest-labs/flux.1-dev", "cocktailpeanut/xulf-d"],
            layer_256=[
                "ad8763121f98e28bc4a3d5a8b494c1e8f385f14abe92fc0ca5e4ab3191f3a881",
                "20d47474da0714979e543b6f21bd12be5b5f721119c4277f364a29e329e931b9",
            ],
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 3.5, "num_inference_steps": 50, "max_sequence_length": 512},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="f-lite-8b",
            repo=["freepik/flux.1-lite-8b"],
            dep_repo=["github.com/fal-ai/f-lite.git"],
            dep_alt={"f_lite": ["FLitePipeline"]},
            gen_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5, "height": 1024, "width": 1024},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="f-lite-7b",
            repo=["freepik/f-lite-7b"],
            dep_repo=["github.com/fal-ai/f-lite.git"],
            dep_alt={"f_lite": ["FLitePipeline"]},
            gen_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5, "height": 1024, "width": 1024},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="lite-texture",
            repo=["freepik/f-lite-texture"],
            dep_repo=["github.com/fal-ai/f-lite.git"],
            dep_alt={"f_lite": ["FLitePipeline"]},
            gen_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5, "height": 1024, "width": 1024},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="hybrid",
            layer_256=[
                "14d0e1b573023deb5a4feaddf85ebca10ab2abf3452c433e2e3ae93acb216443",
                "14d0e1b573023deb5a4feaddf85ebca10ab2abf3452c433e2e3ae93acb216443",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="mini",
            repo=["TencentARC/flux-mini"],
            dep_alt={"diffusers": ["diffusers"]},
            dep_repo=["TencentARC/FluxKits"],
            layer_256=["e4a0d8cf2034da094518ab058da1d4aea14e00d132c6152a266ec196ffef02d0"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="auraflow",
            comp="0",
            dep_pkg={
                "diffusers": ["AuraFlowPipeline"],
            },
            repo=["fal/AuraFlow-v0.3", "fal/AuraFlow-v0.2", "fal/AuraFlow"],
            gen_kwargs={"width": 1536, "height": 768},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="hunyuandit",
            comp="diffusers",
            dep_pkg={"diffusers": ["HunyuanDiTPipeline"]},
            repo=["Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"],
            gen_kwargs={"guidance_scale": 6},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="hunyuandit",
            comp="distilled",
            repo=["Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"],
            gen_kwargs={"num_inference_steps": 25},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="pixart-sigma",
            comp="xl-2-1024",
            dep_pkg={"diffusers": ["PixArtSigmaPipeline"]},
            repo=["PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"],
            init_kwargs={"torch_dtype": "torch.float16", "use_safetensors": True},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="cogview-3",
            comp="plus-3b",
            repo=["THUDM/CogView3-Plus-3B"],
            dep_pkg={"diffusers": ["CogView3PlusPipeline"]},
            gen_kwargs={
                "guidance_scale": 4.0,
            },
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="cogview-4",
            comp="6b",
            repo=["THUDM/CogView4-6B"],
            dep_pkg={"diffusers": ["CogView4Pipeline"]},
            gen_kwargs={"guidance_scale": 3.5, "num_images_per_prompt": 1},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="lumina-image",
            comp="2.0",
            repo=["alpha-vllm/lumina-image-2.0"],
            gen_kwargs={
                "height": 1024,
                "width": 1024,
                "guidance_scale": 4.0,
                "num_inference_steps": 50,
                "cfg_trunc_ratio": 0.25,
                "cfg_normalization": True,
            },
            dep_pkg={"diffusers": ["Lumina2Pipeline"]},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="fuse-dit",
            comp="2b",
            repo=["ooutlierr/fuse-dit"],
            dep_repo=["github.com/tang-bd/fuse-dit.git"],
            dep_pkg={"diffusion": ["pipelines.FuseDiTPipeline"]},
            gen_kwargs={
                "width": 512,
                "height": 512,
                "num_inference_steps": 25,
                "guidance_scale": 6.0,
                "use_cache": True,
            },
        )
    )


def build_mir_art(mir_db: MIRDatabase):
    """Create mir autoregressive info database"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="phi-4",
            comp="multimodal-instruct",
            repo=["microsoft/Phi-4-multimodal-instruct"],
            dep_pkg={"transformers": ["AutoModelForCausalLM"]},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="audiogen",
            comp="medium-1.5b",
            repo=["facebook/audiogen-medium"],
            dep_pkg={"audiocraft": ["models", "AudioGen"]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="parler-tts",
            comp="tiny-v1",
            repo=["parler-tts/parler-tts-tiny-v1"],
            dep_pkg={"parler_tts": ["ParlerTTSForConditionalGeneration"], "transformers": ["AutoTokenizer"]},
            init_kwargs={"AutoTokenizer": {"return_tensors": "pt"}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="parler-tts",
            comp="large-v1",
            repo=["parler-tts/parler-tts-large-v1"],
            dep_pkg={"parler_tts": ["ParlerTTSForConditionalGeneration"], "transformers": ["AutoTokenizer"]},
            init_kwargs={"AutoTokenizer": {"return_tensors": "pt"}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="lumina-mgpt",
            comp="7B-768",
            repo=["Alpha-VLLM/Lumina-mGPT-7B-768"],
            dep_repo=["github.com/Alpha-VLLM/Lumina-mGPT"],
            dep_pkg={"inference_solver": ["FlexARInferenceSolver"]},
            init_kwargs={"precision": "bf16", "target_size": 768},
            gen_kwargs={"images": [], "qas": [["q1", None]], "max_gen_len": 8192, "temperature": 1.0},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="llama-3.1",
            comp="8b-instruct",
            repo=["meta-llama/llama-3.1-8b-instruct"],
            dep_pkg={"transformers": ["AutoModel"]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="orpheus",
            comp="3b-0.1-ft",
            repo=["canopylabs/orpheus-3b-0.1-ft"],
            dep_pkg={"orpheus_tts": ["OrpheusModel"]},
            dep_repo=["github.com/canopyai/Orpheus-TTS"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="t5",
            comp="large",
            repo=["google-t5/t5-large"],
            dep_pkg={"transformers": ["pipeline"]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="outetts-0.3",
            comp="1b",
            repo=["outeai/outetts-0.3-1b"],
            dep_pkg={"outetts": ["InterfaceHF"]},
        )
    )


def build_mir_seq2seq(mir_db: MIRDatabase):
    """Sequence to sequence models"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="seq2seq",
            series="utravox",
            comp="v0_5-llama-3_1-8b",
            repo=["fixie-ai/ultravox-v0_5-llama-3_1-8b"],
            dep_pkg={"transformers": ["pipeline"]},
        )
    )


def build_mir_embedding(mir_db: MIRDatabase):
    """embedding model information"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="embedding",
            series="all-MiniLM",
            comp="L6-v2",
            repo=["sentence-transformers/all-minilm-l6-v2"],
            dep_pkg={"transformers": ["pipeline"]},
        )
    )


def build_mir_mix(mir_db: MIRDatabase):
    """mixed-type architecture"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="mix",
            series="bagel",
            comp="7B-MoT",
            repo=["ByteDance-Seed/BAGEL-7B-MoT"],
            dep_repo=["github.com/ByteDance-Seed/Bagel/"],
        )
    )


def build_mir_lora(mir_db: MIRDatabase):
    """Create mir lora database"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="dmd",
            comp="stable-diffusion-xl",
            repo=["tianweiy/DMD2/"],
            scheduler="ops.scheduler.lcm",
            scheduler_kwargs={},
            dep_pkg={"diffusers": ["diffusers"]},
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0, "timesteps": [999, 749, 499, 249]},
            file_256=[
                "b3d9173815a4b595991c3a7a0e0e63ad821080f314a0b2a3cc31ecd7fcf2cbb8",
                "a374289e9446d7f14d2037c4b3770756b7b52c292142a691377c3c755010a1bb",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="dpo",
            comp="stable-diffusion-xl",
            repo=["radames/sdxl-DPO-LoRA"],
            scheduler="ops.scheduler.dpm",
            scheduler_kwargs={"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True, "order": 2},
            dep_pkg={"diffusers": ["diffusers"]},
            gen_kwargs={"guidance_scale": 7.5, "num_inference_steps": 4},
            file_256=[
                "666f71a833fc41229ec7e8a264fb7b0fcb8bf47a80e366ae7486c18f38ec9fc0",
                "6b1dcbfb234d7b6000948b5b95ccebc8f903450ce2ba1b50bc3456987c9087ad",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp="stable-diffusion-xl",
            repo=["jasperai/flash-sdxl"],
            scheduler="ops.scheduler.lcm",
            dep_pkg={"diffusers": ["diffusers"]},
            scheduler_kwargs={},
            file_256=["afe2ca6e27c4c6087f50ef42772c45d7b0efbc471b76e422492403f9cae724d7"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp="pixart-alpha",
            repo=["jasperai/flash-pixart"],
            file_256=["99ef037fe3c1fb6d6bbefdbb85ad60df434fcc0577d34c768d752d60cf69681b"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp="stable-diffusion-3",
            repo=["jasperai/flash-sd3"],
            file_256=["85fce13c36e3739aa42930f745eb9fceb6c53d53fb17e2a687e3234c1a58ee15"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp="stable-diffusion-1",
            repo=["jasperai/flash-sd"],
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0},
            file_256=["99353444c1a0f40719a1b3037049dbd24800317979a73c312025c05af3574a5f"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            dep_pkg={"diffusers": ["diffusers"]},
            comp="stable-diffusion-xl",
            repo=["ByteDance/Hyper-SD"],
            init_kwargs={"fuse": 1.0},
            file_256={
                "0b97f447b5878323a28fbe7c51ba7acebd21f4d77552ba77b04b11c8911825b6": {"num_inference_steps": 12},
                "55b51334c85061afff5eff7c550b61963c8b8607a5868bbe4f26db49374719b1": {"num_inference_steps": 8},
                "c912df184c5116792d2c604d26c6bc2aa916685f4a793755255cda1c43a3c78a": {"num_inference_steps": 1, "guidance_scale": 0.0},
                "69b25c0187ced301c3603c599c0bc509ac99b8ac34db89a2aecc3d5f77a35187": {"num_inference_steps": 2, "guidance_scale": 0.0},
                "12f81a27d00a751a40d68fd15597091896c5a90f3bd632fb6c475607cbdad76e": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "ca689190e8c46038550384b5675488526cfe5a40d35f82b27acb75c100f417c1": {"num_inference_steps": 8, "guidance_scale": 0.0},
            },
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            comp="flux-1:dev",
            repo=["ByteDance/Hyper-SD"],
            init_kwargs={"fuse": 0.125},
            file_256={
                "6461f67dfc1a967ae60344c3b3f350877149ccab758c273cc37f5e8a87b5842e": {"num_inference_steps": 16, "guidance_scale": 0.0},
                "e0ab0fdf569cd01a382f19bd87681f628879dea7ad51fe5a3799b6c18c7b2d03": {"num_inference_steps": 8, "guidance_scale": 0.0},
            },
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            comp="stable-diffusion-3",
            repo=["ByteDance/Hyper-SD"],
            init_kwargs={"fuse": 0.125},
            file_256={
                "5b4d0b99d58deb811bdbbe521a06f4dbf56a2e9148ff3211c594e0502b656bc9": {"num_inference_steps": 16},
                "0ee4e529abd17b06d4295e3bb91c0d4ddae393afad86b2b43c4f5eeb9e401602": {"num_inference_steps": 4},
                "fc6a3e73e14ed11e21e4820e960d7befcffe7e333850ada9545f239e9aa6027e": {"num_inference_steps": 8},
            },
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            comp="stable-diffusion-1",
            repo=["ByteDance/Hyper-SD"],
            file_256={
                "64b98437383537cd968fda6f87a05c33160ece9c79ff4757949a1e212ff78361": {"num_inference_steps": 12},
                "f6123d5b950d5250ab6c33600e27f4dcf71b3099ebf888685e01e9e8117ce482": {"num_inference_steps": 8},
                "a04fd9a535c1e56d38f7590ee72a13fd5ca0409853b4fff021e5a9482cf1ca3b": {"num_inference_steps": 1, "guidance_scale": 0.0},
                "2f26dcc1d883feb07557a552315baae2ca2a04ac08556b08a355a244547e8c3a": {"num_inference_steps": 2, "guidance_scale": 0.0},
                "c5dd058616461ed5053e2b14eec4dbe3fa0eea3b13688642f6d6c80ea2ba5958": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "91fc3186236e956d64dbb4357f2e120c69b968b78af7d2db9884a5ca74d3cd13": {"num_inference_steps": 8, "guidance_scale": 0.0},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="stable-diffusion-xl",
            repo=["latent-consistency/lcm-lora-sdxl"],
            init_kwargs={"fuse": 1.0},
            gen_kwargs={
                "num_inference_steps": 8,
            },
            dep_pkg={"diffusers": ["diffusers"]},
            scheduler="ops.scheduler.lcm",
            scheduler_kwargs={"timestep_spacing": "trailing"},
            file_256=["a764e6859b6e04047cd761c08ff0cee96413a8e004c9f07707530cd776b19141"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="ssd-1b",
            repo=["latent-consistency/lcm-lora-ssd-1b"],
            gen_kwargs={"num_inference_steps": 8},
            file_256=["7adaaa69db6f011058a19fd1d5315fdf19ef79fcd513cdab30e173833fd5c59b"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="vega",
            repo=["segmind/Segmind-VegaRT"],
            gen_kwargs={"num_inference_steps": 8},
            file_256=["9b6e8cd833fa205eaeeed391ca623a6f2546e447470bd1c5dcce3fa8d2f26afb"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="stable-diffusion-1",
            repo=["latent-consistency/lcm-lora-sdv1-5"],
            gen_kwargs={"num_inference_steps": 8},
            file_256=["8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4", "eaecb24a1cda4411eab67275b1d991071216ac93693e8fa0c9226c9df0386232"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lightning",
            comp="stable-diffusion-xl",
            repo=["ByteDance/SDXL-Lightning"],
            dep_pkg={"diffusers": ["diffusers"]},
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="pcm",
            comp="stable-diffusion-xl",
            repo=["wangfuyun/PCM_Weights"],
            file_256={
                "0365f6107250a4fed1b83e8ae6a070065e026a2ba54bff65f55a50284232bbe6": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "04ea827435d5750e63d113dc509174b4f6e8a069ff8f91970c3d25299c10b1f8": {"num_inference_steps": 16},
                "7eb353b2abcaabab6251ba4e17d6cbe2e763feb0674b0f950555552212b44621": {"num_inference_steps": 16},
                "a85cf70ac16ed42011630a5cd6b5927722cb7c40a2107eff85e2670f9a38c893": {"num_inference_steps": 4},
                "9f7f13bb019925eacd89aeff678e4fd831f7b60245b986855dff6634aee4eba9": {"num_inference_steps": 4},
                "3b9c970a3e4c0e182931e71b3f769c1956f16c6b06db98b4d67236790d4d0b1d": {"num_inference_steps": 8},
                "7f04ba8911b4c25ef2c7cbf74abcb6daa3b4f0e4bc6a03896bdae7601f2f180b": {"num_inference_steps": 8},
                "13fb038025ce9dad93b8ee1b67fc81bac8affb59a77b67d408d286e0b0365a1d": {"num_inference_steps": 16, "guidance_scale": 0.0},
                "3442eff271aa3b60a094fd6f9169d03e49e4051044a974f6fcf690507959191f": {"num_inference_steps": 16, "guidance_scale": 0.0},
                "242cbe4695fe3f2e248faa71cf53f2ccbf248a316973e4b2f38ab9e34f35a5ab": {"num_inference_steps": 2, "guidance_scale": 0.0},
                "e1f600491bb8e0cd94f41144321e44fdb2cb346447f31e71f6e53f1c24cccfbf": {"num_inference_steps": 2, "guidance_scale": 0.0},
                "d0bf40a7f280829195563486bec7253f043a06b1f218602b20901c367641023e": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "212150d7953627fb89df99aad579d6763645a1cb2ef26b19fee8b398d5e5ff4d": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "e80fcf46d15f4d3821d3d9611bdb3022a4a8b647b2536833b168d317a91e4f74": {"num_inference_steps": 8, "guidance_scale": 0.0},
                "56ed9dc9f51f4bb0d6172e13b7947f215c347fc0da341c8951b2c12b9507d09e": {"num_inference_steps": 8, "guidance_scale": 0.0},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="pcm",
            comp="stable-diffusion-1",
            repo=["wangfuyun/PCM_Weights"],
            file_256={
                "b80b27dd6504f1c3a7637237dda86bc7e26fa5766da30c4fc853c0a1d46bad31": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "8f605ffde3616592deb37ed8c6bacb83fe98963c1fd0883c2a4f93787098aa45": {"num_inference_steps": 16},
                "fa6acb94f11dba3bf4120af5a12e3c88cd2b9572d43ec1a6fb04eede9f32829e": {"num_inference_steps": 4},
                "bff3d4499718b61455b0757b5f8d98fe23e73a768b538c82ecf91c693b69dbcd": {"num_inference_steps": 8},
                "c7ac2fa3df3a5b7080ebe63f259ab13630014f104c93c3c706d77b05cc48506b": {"num_inference_steps": 16, "guidance_scale": 0.0},
                "4c5f27a727d12146de4b1d987cee3343bca89b085d12b03c45297af05ce88ef4": {"num_inference_steps": 2, "guidance_scale": 0.0},
                "29278bc86274fdfc840961e3c250758ff5e2dc4666d940f103e78630d5b879d3": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "41a7f0b966d18f643d16c4401f0b5ef6b9ef7362c20e17128322f17874709107": {"num_inference_steps": 8, "guidance_scale": 0.0},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="pcm",
            comp="stable-diffusion-3",
            repo=["wangfuyun/PCM_Weights"],
            file_256={
                "8a45878ecc34e53855fe21146cb6ef32682053b7c4eacc013be89fb08c4c19d8": {"num_inference_steps": 2, "guidance_scale": 1.2},
                "9444a5cead551c56c4d1c455ce829ba9f96f01fbcca31294277e0862a6a15b76": {"num_inference_steps": 4, "guidance_scale": 1.2},
                "e365902c208cbc0456ca5e7c41a490f637c15f3f7b98691cbba21f96a8c960b4": {"num_inference_steps": 4, "guidance_scale": 1.2},
                "3550fa018cd0b60d9e36ac94c31b30f27e402d3855ed63e47668bb181b35a0ad": {"num_inference_steps": 4, "guidance_scale": 1.2},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="slam",
            comp="stable-diffusion-xl",
            repo=["alimama-creative/slam-lora-sdxl/"],
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 1},
            dep_pkg={"diffusers": ["diffusers"]},
            scheduler="ops.scheduler.lcm",
            scheduler_kwargs={"timestep_spacing": "trailing"},
            file_256=["22569a946b0db645aa3b8eb782c674c8e726a7cc0d655887c21fecf6dfe6ad91"],
        )
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="slam", comp="stable-diffusion-1", repo=["alimama-creative/slam-sd1.5"]))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp="stable-diffusion-xl",
            repo=["SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA"],
            dep_pkg={"diffusers": ["diffusers"]},
            gen_kwargs={"guidance_scale": 5.0},
            file_256=["0b9896f30d29daa5eedcfc9e7ad03304df6efc5114508f6ca9c328c0b4f057df"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp="stable-diffusion-1",
            repo=["SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep_LoRA"],
            gen_kwargs={"guidance_scale": 7.5},
            file_256=["1be130c5be2de0beacadd3bf0bafe3bedd7e7a380729932a1e369fb29efa86f4"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="tcd",
            comp="stable-diffusion-xl",
            repo=["h1t/TCD-SDXL-LoRA"],
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0, "eta": 0.3},
            dep_pkg={"diffusers": ["diffusers"]},
            scheduler="ops.scheduler.tcd",
            scheduler_kwargs={},
            file_256=["2c777bc60abf41d3eb0fe405d23d73c280a020eea5adf97a82a141592c33feba"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="tcd",
            comp="stable-diffusion-1",
            repo=["h1t/TCD-SD15-LoRA"],
            file_256=["eaecb24a1cda4411eab67275b1d991071216ac93693e8fa0c9226c9df0386232"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp="flux-1",
            repo=["alimama-creative/FLUX.1-Turbo-Alpha"],
            dep_pkg={"diffusers": ["diffusers"]},
            gen_kwargs={"guidance_scale": 3.5, "num_inference_steps": 8, "max_sequence_length": 512},
            init_kwargs={"fuse": 0.125},
            file_256=["77f7523a5e9c3da6cfc730c6b07461129fa52997ea06168e9ed5312228aa0bff"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp="stable-diffusion-3.5-medium",
            repo=["tensorart/stable-diffusion-3.5-medium-turbo"],
            scheduler=["ops.scheduler.flow-match"],
            scheduler_kwargs={"shift": 5},
            init_kwargs={"fuse": 1.0},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp="stable-diffusion-3.5-large",
            repo=["tensorart/stable-diffusion-3.5-large-TurboX"],
            scheduler=["ops.scheduler.flow-match"],
            scheduler_kwargs={"shift": 5},
            init_kwargs={"fuse": 1.0},
        )
    )


def build_mir_other(mir_db: MIRDatabase):
    """Create mir info database"""
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="patch",
            series="hidiffusion",
            comp="stable-diffusion-xl",
            num_inference_steps=10,
            timesteps="StableDiffusionXLTimesteps",
            dep_pkg={"hidiffusion": ["apply_hidiffusion"]},
            repo=["github.com/megvii-research/HiDiffusion/"],
            gen_kwargs={"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5},
        )
    )


def build_mir_dtype(mir_db: MIRDatabase):
    """Create mir info database"""
    import torch
    from nnll.metadata.helpers import slice_number

    available_dtypes = [dtype for dtype in torch.__dict__.values() if isinstance(dtype, torch.dtype)]
    for precision in available_dtypes:
        dep_name, class_name = str(precision).split(".")
        if "_" in class_name:
            series_name = class_name[0].upper() + class_name.split("_")[1].upper()
        else:
            series_name = class_name[0].upper() + str(slice_number(class_name))
        variant_name = class_name.replace("float", "fp")
        mir_db.add(
            mir_entry(
                domain="ops",
                arch="precision",
                series=series_name,
                comp="[init]",
                variant=variant_name,
                package={dep_name.lower(): class_name.lower()},
            )
        )


def build_mir_scheduler(mir_db: MIRDatabase):
    """Create mir info database"""
    try:
        from diffusers import _import_structure

        for series_name in _import_structure["schedulers"]:
            class_name = series_name
            for minor in ["Discrete", "Scheduler", "Multistep", "Solver"]:
                series_name = series_name.replace(minor, "")
            series_name.lower()
            mir_db.add(
                mir_entry(
                    domain="ops",
                    arch="scheduler",
                    series=series_name,
                    comp="[init]",
                    requires={"diffusers": class_name},
                )
            )
    except (ImportError, ModuleNotFoundError) as error_log:
        dbug(error_log)

    mir_db.add(
        mir_entry(
            domain="ops",
            arch="scheduler",
            series="align-your-steps",
            comp="stable-diffusion-xl",
            num_inference_steps=10,
            timesteps="StableDiffusionXLTimesteps",
            dep_alt={"diffusers": ["schedulers.scheduling_utils", "AysSchedules"]},
        )
    )


def main(mir_db: Callable = MIRDatabase()) -> None:
    """Build the database"""
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # root_path = os.path.dirname(current_dir)
    # sys.path.append(root_path)
    # sys.path.append(os.getcwd())
    # print(sys.path)

    build_mir_unet(mir_db)
    build_mir_dit(mir_db)
    build_mir_art(mir_db)
    build_mir_seq2seq(mir_db)
    build_mir_embedding(mir_db)
    build_mir_mix(mir_db)
    build_mir_lora(mir_db)
    build_mir_scheduler(mir_db)
    build_mir_dtype(mir_db)
    build_mir_other(mir_db)
    mir_db.write_to_disk()


# if __name__ == "__main__":
# import sys

# sys.path.append(os.getcwd())
# main()
