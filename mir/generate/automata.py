# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""自動化索引"""
# regex to remove \[[^\]]*\]
# (?m)^\s*"[^"]+"(?=\s*:)
# (?m)^\s*"[^"]+"\s?:
# modelspec sai https://github.com/Stability-AI/ModelSpec

from importlib import import_module
import re
from typing import Dict, List, Tuple, Any

import torch

from mir.indexers import diffusers_index, transformers_index
from mir.maid import MIRDatabase
from mir.spec import mir_entry
from mir.tag import tag_model_from_repo, tag_scheduler, tag_base_model, tag_pipe


sd1_series, sd1_comp = tag_model_from_repo("stable-diffusion-v1-5/stable-diffusion-v1-5")
sdxl_series, sdxl_comp = tag_model_from_repo("stabilityai/stable-diffusion-xl-base-1.0")
dev_series, dev_comp = tag_model_from_repo("black-forest-labs/FLUX.1-dev")
schnell_series, schnell_comp = tag_model_from_repo("black-forest-labs/FLUX.1-schnell")
ssd_series, ssd_comp = tag_model_from_repo("segmind/SSD-1B")
vega_series, vega_comp = tag_model_from_repo("segmind/Segmind-Vega")
sd3_series, sd3_comp = tag_model_from_repo("stable-diffusion-3.5-medium")  #


# def auto_gan etc etc
# ai-forever/Real-ESRGAN


def add_mir_diffusion(mir_db: MIRDatabase):
    """Create MIR entries missing from the database"""

    repo = "microsoft/speecht5_hifigan"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="gan",
            series=series,
            comp=comp,
            file_256=[
                "d9dc6513c30a5b86c2497712690c04fe74b4aa79fdab6d490b34fcb4e24c590c",
            ],
            layer_b3=[
                "85b5acdf29ad04c63f885383340d8e3445ae0055521f82cabb82bd09cfb9a956",
            ],
            layer_256=[
                "bd52b538e7ac05711be9321cfb7619d4056996ce32923c9c91ee02cf69154770",
            ],
        )
    )
    series, comp = tag_model_from_repo("lodestones/Chroma")
    repo = "lodestones/Chroma1-HD"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={
                "0": {
                    # "diffusers": "ChromaPipeline",
                    "generation": {
                        "num_inference_steps": 40,
                        # "guidance_scale": 3.0,
                        # "num_images_per_prompt": 1,
                    },
                }
            },
            file_256=[
                "d845553f11e6afe8139c41ca73678f9f03eab2e68d2e1c6f03ae19509a4d546",  # sai
                "1b2993a44e63b2250496f69edce643bac2fb79833cf92ba8dd95cbd764d970c7",  # annealed sai
                "2dd46f08516246df1f582047cc09268ce4f747357baff05b13148e71519029fc",  # diffusers
            ],
            # layer_b3=[
            # "8da38c3719e77a38a20356c9f92f5ca0101c17406d7a9817323cf67b74088520",  # diffusers
            # ],
            # layer_256=[
            # "267798815e0855c2253061c6a6ab70edf9590e8ea1ba9b4621eeb0f6615ee37b",
            # ],
        )
    )
    repo = "lodestones/Chroma1-Flash"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={
                "0": {
                    "diffusers": "ChromaPipeline",
                    "generation": {
                        "num_inference_steps": 8,
                        "guidance_scale": 1.0,
                        "num_images_per_prompt": 1,
                    },
                },
            },
            file_256=[
                "2c0c7d908d04418a48b453c293237a9826d54472cf0ba76e28697d1309d1021b",  # sai
                "c88f6794753ba23e8f6bf8c84cf220daa35a6aa16d54ea0c3e0136f52e5da7e1",  # sai delta
                "c759d67ca3ef50a9a1c242e3291c57f406646f226a95f43f66577996494986db",  # diffusers
            ],
            # layer_b3= [""],
            # "layer_256"= [""],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp="pony-diffusion",
            file_256=["67ab2fd8ec439a89b3fedb15cc65f54336af163c7eb5e4f2acc98f090a29b0b3"],
            layer_b3=["bf4c2154daa4ece7292277b210d081f98759e9ed4d5c889564632e3ccc4a1071"],
            layer_256=["465425d4420dcf5aa4b4d5b456db11a1fcc7c8f61b2e4a87e2470297c98bb96e"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp="pony-diffusion-turbo",
            file_256=[
                "7555ac941f3a767833830ba5cc9a4508a9777cbf97b487b6baf0400ab7000587",  # turbomerge
                "9322f9d91b28abf09e4137bc02ec806af23510221a164e71b81778e61cc3b4b2",  # turbosimple
            ],
            layer_b3=[
                "1e8f23fcd4be0f00eb52368b91c709fffa8a3b8e21772b92b2e0671eed9117d0",
                "5c8b3f34f9d0a58135cf72fbfe9b5d75b5545a10e3d726478543fa7cc510a8bc",
            ],
            layer_256=[
                "7edf51ef09b39c46937a4e4141707c040cd12af0d95299a4d3cd2b7d3fabe035",
                "74e4dbc89d57d61ff7e8af8b0fddcf7466ba233d53ca4ffb7777138991bc3d52",
            ],
        )
    )
    repo = "cagliostrolab/animagine-xl-4.0"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            file_256=[
                "8ece83aa1bed1fb39a2b81f1660f0ce6889218e493c1f2ed55e9f15f59a7e03f",  # v4
                "6327eca98bfb6538dd7a4edce22484a1bbc57a8cff6b11d075d40da1afb847ac",  # v4 opt
                "1449e5b0b9de87b0f414c5f29cb11ce3b3dc61fa2b320e784c9441720bf7b766",  # v3
                "e3c47aedb06418c6c331443cd89f2b3b3b34b7ed2102a3d4c4408a8d35aad6b0",  # v3.1
            ],
            layer_b3=[
                "268ffbb120670b9c4b25158bd474c787740884b7738b48203aa03c4c3f00028f",
                "18fda1a55cad137d62c81d4328f5ece85d88b126261e06b9e14ab68055d5d484",
                "bae9bc8a5c43145bcf92ee3391618d9eaddd689f626991bae202de9cf5f1e70e",
                "d6bc5ccafa2b97c867b13a1e7a8c2c7ad9c4877055a66c71bb773557bc306447",
            ],
            layer_256=[
                "c21d1c38813e078817122e12866ab39f5aa7f56945dd4a8beee3cae1e0f139e7",
                "b916c162c981155aaf74e93d5314038af6767bb5a129c51ee05a1fb6a206c6ac",
                "ecc6bfc73824a2d7c3b0ca184854a235859f329c83768f017b07a19a535d17b4",
                "97f6ca05de7fbdae7aacb2427a552f924492176c474a23dd252c192e1c0e9d65",
            ],
        )
    )
    repo = "OnomaAIResearch/Illustrious-XL-v2.0"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            file_256=[
                "c2a1a3eaa13d4c107dc7e00c3fe830cab427aa026362740ea094745b3422a331",  # v2
                "536863e9f0c13b0ce834e2f8a19ada425ee4f722c0ad3d0051ec7e6adaa8156c",  # 1.1
                "3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9",  # 0.1
                "e3d12d0f76d61aa31d2668a2217e5b642592193f2946842c44d7056ea5469cce",  # 0.1 guided
                "735cf3fefcbdc4f7817f53247e38b836ffd27c7641af6d8daa21d245242cb4bd",  # 1.0
            ],
            layer_b3=[
                "93b061baf21d743d592327a61f027d099d8e18da9808a76c7704ad123eba4a29",
                "dc05fed2acbc73cef4c377cfa2a681c5cf6d065b88d8bf70d371bbcce6a223a8",
                "8eb1c30327e5b71b35b9a4513dc5f2cac9f244667393c0eedb10a26aa9991cd8",
                "3dafbe31f6ebaffa3d054e1b37049e1147faa2474ceb6dab7bc3c4cded0c845e",
                "892533778ee14454938f7b50830093f58e12f1e14560a148f71927e4ccff5f5c",
            ],
            layer_256=[
                "397791b3d77affb7bd35c5ded7377493c6bf456920a41388ba95bd0157109803",
                "b23c02b8519c6777a1f271662f4251a59468c4b3e11184a2d722fa8929b4ea48",
                "a373981494f5508c124a1960bdd096bbc96935fbb54b1218f563206d3892c176",
                "b709df257c40d9d981f686f2880bbe64f43b78805b7213768d659a142a593efd",
                "f1e6b4cab0fce608dca6fa851384e8728202449f16270fbd1f0c4c5ec4946c10",
            ],
        )
    )
    repo = "playgroundai/playground-v2.5-1024px-aesthetic"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            file_256=[
                "11b6d7bce65674659cc6b7ea960658436edfd80e566cb240ebd4bfbc3e2076c8",  # 2.5 diffusers
                "bcaa7dd6780974f000b17b5a6c63e6f867a75c51ffa85c67d6b196882c69b992",  # 2.5 aes sai fp16
                "956dca99114aaa5c3eb526381309d37ee96737e78ed64c8ae613409f47c3f65a",  # 2.5 aes sai
                "933778ce76c1fc0ca918b37e1488411b8a99bbd3279c12f527a3ac995a340864",  # 2.5 fp16 diffusers
                "5c7d38880d0940e6795158b7608ccef89217272b1f2a9331c5b0a2adffcd82c4",  # v2 sai
                "0411e988479884b1a3ecd184123efe38d051d8d0ef24270585a7d1d57499464a",  # v2 sai fp16
            ],
            layer_b3=[
                "d55b22740da2d5b98020ad2390cdc0a7ee08cf9e0d98c11957f16cc20c49815b",  # 2.5 diffusers
                "7e9be9bd9a3aed1ad7207e2f77c98c24c3a75f6adcc9b53514033c6c3365d289",  # 2.5 aes sai fp16
                "5c6dfcc8d01dfb64723f8f5785caa080e2987859c0a050470bfdbe5312be9efc",  # 2.5 aes sai
                "703f775c6e48ed5b0eba6e847414f047bcd4adc677dbc1bf221b3ef05b2ac471",  # 2.5 diffusers fp16
                "72d4ebe4af61f8a7add8fe36b8acd16602894279fb5a744ad50b5b5bac7067b8",  # v2 sai
                "acb757b851db12cdf9d4365a45ee0d6e64afa77ac95583bb82711baf7c4125fd",  # v2 sai fp16
            ],
            layer_256=[
                "adb7be228d4ee6e583c3e5ae4ddb579fef64c3987617ce4d4aff3eb7f8d6a3f7",
                "d4813e9f984aa76cb4ac9bf0972d55442923292d276e97e95cb2f49a57227843",  # 2.5 aes sai fp16
                "fe2e9edf7e3923a80e64c2552139d8bae926cc3b028ca4773573a6ba60e67c20",
                "bc7021473a04a6de3fe0d0fed600875d852ad1ad9d47c445278f66ce9e8ec7a0"  # 2.5 fp16 diffusers
                "fc94481f0c52b21c5ac1fdade8d9c5b210f7239253f86ef21e6198fe393ed60e",  # v2 sai
                "a6f31493ceeb51c88c5239188b9078dc64ba66d3fc5958ad48c119115b06120c",  # v2 sai fp16
            ],
            pkg={
                0: {
                    "diffusers": "DiffusionPipeline",
                    "precision": "ops.precision.float.F16",
                    "generation": {"num_inference_steps": 50, "guidance_scale": 3},
                }
            },
            identifiers=[
                "edm_mean",
                [1, 4, 1, 1],
                2516,
            ],
        )
    )
    repo = "segmind/Segmind-Vega"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            file_256=[
                "94762e983e5942056be73c5c1d4464b8ffa1ada500b4fef1267550e2447953ce",  # modelspec sai
                "1ab33e37fbb2566c55cd729e4ab79cc2f99cd9d0a578fabc7a2cf4ee47968be1",  # diffusers
                "8cfa375669b1222d6fecf470f41b2abb370c76a90ab9568964c4bb15b34ec8a2",  # diffusers fp16
            ],
            layer_b3=[
                "2f353c5e6ed0a2c05af00d014e18e65f69f1ce8c48f8eefbf8ad71b34f940fbf",
                "cc34bd3135d7cafc3cb6e3f6e7cb6896c98277bad52877a952ddbd2ffe222e01",
                "b90efdc848f5386d5250b6fb233ce380cf6cc299f497cfa1d2feaef22f87c9d1",
            ],
            layer_256=[
                "029b89ee311110c8f945dbdfc52c1d5daeb1e78c353c38aa3141ec68ce28e7cc",
                "5cdb948e5f3873300679073391d48fc648171f02093d7737d078557ff75762bb",
                "f73afbe43cc76571cb86ebcfced618668a2fb2252b0bc6ba88d6e942bae75741",
            ],
        )
    )
    repo = "segmind/SSD-1B"

    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            file_256=[
                "7cb406ec0662e91570a79f3c4fb8f0ea5325bffe6af5d9382edae838698f72bd",  # modelspec sai
                "1895a00bfc769a00b0c0c43a95e433e79e9db8a85402b45a33e8448785bde94d",  # a1111 aio
                "0bf1ce6b065a6b969ab02dc8e8fa21eb20ee189b10935c49ce68c77a7e432c1c",
                "02ed8ebd0ed55aec686fcf20946d7a1659a31f9f8d9c3798cd254ba6b67434ca",  # diffusers
                "40d8ea9159f3e875278dacc7879442d58c45850cf13c62f5e26681061c51829a",  # diffusers fp16
            ],
            layer_b3=[
                "c074dc38e8ec836816b91cbcc2ca17f80d6106de8d196d416ef9a27c8837ee45",  # modelspec sai
                "1d6c0216da57fe98e7ad29e9653566725f5b2a87845fdbdcda257b3be817b5f4",  # a1111 aio
                "c074dc38e8ec836816b91cbcc2ca17f80d6106de8d196d416ef9a27c8837ee45",
                "89f86d9c846495870416b4945b6a46a517f28405e5bab666feb4057f012340be",
                "535b47e9b70da6494878ca6d45af3f2e201b7f17748432911c12232e586855e6",
            ],
            layer_256=[
                "52267d5d327a2ba92c7a14261a9d081df621b8366819b1bb3a47d130523a813c",
                "b365a3631c6c74532f3a571c84c68e088be35496d35be1e932031713ddd2a2f4",
                "52267d5d327a2ba92c7a14261a9d081df621b8366819b1bb3a47d130523a813c",
                "89f86d9c846495870416b4945b6a46a517f28405e5bab666feb4057f012340be",
                "535b47e9b70da6494878ca6d45af3f2e201b7f17748432911c12232e586855e6",
            ],
        )
    )
    repo = "shuttleai/shuttle-3.1-aesthetic"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=schnell_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={
                2: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 4},
                }
            },
            file_256=[
                "176871da1d5d2d511a52ae9b0dd70faa1f5d1b7734b7e33ed6b4bffa52050e0d",
                "4b80d37681eaed07b7f5b3825a392da929d1620933ede7c2749ef3613cc53f42",
            ],
            layer_b3=[
                "ff422d1734abf33366e87bbf44267dc6096c5d499e695287c35558174877412e",
                "5ad8034eac6b82d842311437101c52b5d35826ce34994940d9e667e702a0d45c",
            ],
            layer_256=[
                "e5d95de314cbfc49b79479118a1ac0b90fc95ccd6bb1a5c95803996d6cebf8fe",
                "d299e8ea4a605917ab98a4a7330d4d398b4ae295efbf458eeeceb5ff1bd7959a",
            ],
        )
    )
    repo = "shuttleai/shuttle-3-diffusion"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=schnell_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={
                2: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 4},
                }
            },
            file_256=[
                "a5b04df4072698395387c21e8da0176d03f6557e0c38ff1dd3bf469ebab9d0fd",  # fp8
                "a91b46de2055b3511ee87523b57862648856e8c00100161d5b520543a7302755",  # norm
                "23a77c86189d5934da48bf44bb871cf80ba99177ffd3fd5272cdecb208c8b8be",  # mlx q8
                "d3782d5a8f6e82c6676e8e26d54020934ada589d2aceb17fc5ca604b1bd55da8",  # mlx q4
            ],
            layer_b3=[
                "4dd3174edf6b680ce9daf3de643e33ae2c4f09a4d5968da61ea48885f3a193c0",
                "9fdf191b2c58b2a6e190396e12314530593dca4f2a2bee389ec5175da5e52af8",
                "ad203ad6a00d8b1315337e34069e7c41016ea407469a536de8ad6807042017fd",
            ],
            layer_256=[
                "14d0e1b573023deb5a4feaddf85ebca10ab2abf3452c433e2e3ae93acb216443",
                "7ce8d449b32a9c959431ade729b513ee7a6457f11e1c13e3ef04dd8db3494621",
                "9c3395f67a3d844483b77f0ddd5e2ea64b61732fa9d9da19845bb8ae574c1f8c",
            ],
        )
    )
    repo = "enhanceaiteam/Mystic"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={0: {"generation": {"num_inference_steps": 16, "guidance_scale": 7.5, "width": 768, "height": 1024}}},
            file_256=[
                "179d4000e44295f6dfadc0e4ac210146454724d46371b82657200ff9fb5c68a9",  # mlx 0
                "48ca85274e3b67f07f70dd84b67725e62395c2f7b188394342716f783ea4c6ac",  # mlx q8
            ],
            layer_b3=[
                "91074aaebe1b5f3b2e7755d3c092af7eb240e92a192360690f1033949d3c8a68",  # mlx 0
            ],
            layer_256=[
                "3942e6a52dbb0abaf63b031d9c4eda0df47576b51d4c81361978a3dc27b1309e",  # mlx 0
            ],
        )
    )
    repo = "shuttleai/shuttle-jaguar"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=schnell_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={
                2: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 4},
                }
            },
            file_256=[
                "dcbc4f2470b177eed12c7d7515c0e7342515a849ebd31a50c8d8d43913d7bd32",
                "26a7aa64c0798a3549e1d767932da0a7fb82b49f8edcbdcde804a20d9ed1478f",  # mlx q8
            ],
            layer_b3=[
                "9906c29933d0c33a6ee8d9712f33fa8bd4b35b46a1c7b565ae48832b757dd980",
                "89c453c4bf99220405687eed984dace4492bdae1b6fb08f3d9629145b1a11672",  # mlx q8
            ],
            sha_256=[
                "4eacf27e5659f5dc42f34c407cbe9e1e202290692df754eb68fe913f59fa2941",
            ],
        )
    )
    repo = "freepik/flux.1-lite-8b"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={0: {"generation": {"num_inference_steps": 28}}},
            file_256=[
                "09e970a7b8d1813ea7cacd48f9a944fd223882b137a8f4f3b61d864cdc20bbec",  # mlx q8
                "de90e69945c2f4afcb9b6a057ce48190905c984370fce76b16ba3b97d46e2747",  # mlx q4
            ],
            layer_b3=[
                "9276fa4805efeb45c08cca32c5b51d490e57a2ce5c15ef476a8e468a509c5cdf",
            ],
            layer_256=[
                "e1afe2f9b1ca55b3c659293cf3237f6b5571f5c4e826bad025ff0f7b54dc34ee",
            ],
        )
    )
    repo = "freepik/f-lite-7b"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={0: {"f_lite": "FLitePipeline", "generation": {"num_inference_steps": 28}}},
        )
    )
    repo = "freepik/f-lite-texture"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={0: {"f_lite": "FLitePipeline", "generation": {"num_inference_steps": 28}}},
        )
    )
    repo = "freepik/f-lite"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={0: {"f_lite": "FLitePipeline", "generation": {"num_inference_steps": 28}}},
        )
    )
    repo = "TencentARC/flux-mini"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            file_256=["4236455adeaeb4ed444d63b253ec99805022d17e962ed7261ada9c72ce11cfee"],
            layer_b3=["c1a6f83585398fe452d20596a79a522e2986f4c2c01a40e7bfd787af113735d3"],
            layer_256=["e4a0d8cf2034da094518ab058da1d4aea14e00d132c6152a266ec196ffef02d0"],
        ),
    )
    repo = "ostris/Flex.2-preview"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            file_256=[
                "0407108e446a4f57efffc5e7518bc374876af970d3c6068dc4074de0d221c615",  # modelspec sai
                "df168ba94d5f96c478b24604a6beedff6189047152190509c73c162ea0d8ec02",  # mlx
            ],
            layer_b3=[
                "7f85cdc186896da6965b57d5edb672f08663075d2b207f0e20e328c4034a8076",  # mlx
            ],
            layer_256=[
                "5063de856be5365807d12b47ef6919b4ac611a72651739b2b4050e113bed7a83"  # mlx,
            ],
        ),
    )
    repo = "ostris/Flex.1-alpha"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            file_256=[
                "5d6dce30a266ccbf530c3a3bf253cd5486720a8fb71cdeed556c28304201dc2f",  # modelspec sai
                "7acf8771b80a91eaa21566abe8c7d9d3ba33d8688e6e98446827749aee7ca1ee",  # mlx
            ],
            layer_b3=[
                "cb3d3edafd81651eefd62894b3572deb02c5304f4b5d4f7ab8654f1fb922ecd6",  # mlx
            ],
            layer_256=[
                "a6b9af6efc25fa77cd24046b81ee66fea09a9987d2a8e56ffca9b7a1c9c9c519"  # mlx,
            ],
        ),
    )
    repo = "tensorart/stable-diffusion-3.5-medium-turbo"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=sd3_series,
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            pkg={
                0: {
                    "precision": "ops.precision.bfloat.B16",
                    "generation": {"num_inference_steps": 8, "guidance_scale": 1.5, "height": 1024, "width": 768},
                }
            },
            file_256=[
                "5b0530e8d71b49fa1358f1208047cd789a40bae5b44406c9524b0f0d88f8b246",  # diffusers
                "07119c77c3548a1d9eb30923df4dd55ec74914dc5ec81626804dcbe51ce17a5d",  # sai
                "3c379381344d2a2b3ee3d7a1bc97f7d1e58fa95c6b5187fb48b3ce446f99f17b",  # q4km gguf
                "6b3806cafdb4303ea2638e9e08eb186067b4a46a95ddf344ccdbe56537afaf6e",  # q8km gguf
            ],
            layer_b3=[
                "873821614080a98e1ebfe56673bc96c2ac57379720d4ad2f97e4bca317571d48",  # diffusers
                "7284d2027523482af9ef47405667ca891cc518bfb6ebf1f1d4666cb0accc8cd5",
                "d938ee5738c73f701760ed18acad274b074d2796123aee3f2eee1328b6c36ea4",
                "c4c40056c2a77959083b5a69a1a4b205caa463ccabde057352c5c4e38b2c67b6",
            ],
            layer_256=[
                "3c324055a1ec6eb4ee0242e344bb2b6356afcbd2e215fdd9d160cda691a72fae",
                "7284d2027523482af9ef47405667ca891cc518bfb6ebf1f1d4666cb0accc8cd5",
                "d938ee5738c73f701760ed18acad274b074d2796123aee3f2eee1328b6c36ea4",
                "c4c40056c2a77959083b5a69a1a4b205caa463ccabde057352c5c4e38b2c67b6",
            ],
        ),
    )
    repo = "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=series,
            comp=comp,
            repo=repo,
            file_256=[
                "",
                "",
            ],
            layer_b3=[
                "",
            ],
            layer_256=[""],
        ),
    )
    repo = "OnomaAIResearch/Illustrious-Lumina-v0.03"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=tag_model_from_repo("Alpha-VLLM/Lumina-Image-2.0")[0],
            comp=tag_model_from_repo(repo)[0],
            repo=repo,
            file_256=[
                "dc6cffcfb0ccfca6332ddb5d2fe25bcb5f496f44b481627f48c42626156fa6a8",  # 2b 22100 ema unified fp32
                "2ac549741fa1c6de2d6cd8be06abcdce52d472eeae2439f948e285258b66a214",  # 0.03 ema
            ],
            layer_b3=[
                "a97b4a63e1e7678e8e7154fae55252267bd1f0ba76b03dba622d801644e657ac",
                "aa6c1b2d1971cea3c4ed0963c8d68d4c50db683f8eab9f77f60ea2d04ed6ce5c",
            ],
            layer_256=[
                "39086c199b9ac296dcba53461ba1e113906d91fbc1b12556d92f5cc77ca11f9f",
                "e51ba2ded40f1af5ca6f78c46eed8305fbd87cd6401e9d439837e10d35cc5828",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="patch",
            series="hidiffusion",
            comp=sdxl_series,
            pkg={
                0: {
                    "hidiffusion": {"apply_hidiffusion": {"timesteps": "StableDiffusionXLTimesteps"}},
                    "generation": {"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5, "num_inference_steps": 10},
                },
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="scheduler",
            series="align-your-steps",
            comp=sdxl_series,
            pkg={
                0: {
                    "diffusers": "schedulers.scheduling_utils.AysSchedules",
                    "generation": {"timesteps": "StableDiffusionXLTimesteps", "num_inference_steps": 10},
                }
            },
        )
    )
    # possible mixed-type architecture?
    # fusion / united / universal


def add_mir_llm(mir_db: MIRDatabase):
    base_arch, base_series, base_comp = tag_base_model(repo_path="facebook/chameleon-7b", class_name="ChameleonModel")
    repo = "Alpha-VLLM/Lumina-mGPT-7B-1024"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=base_series,
            comp=series,
            repo=repo,
            pkg={
                0: {
                    "inference_solver": {"FlexARInferenceSolver": {"precision": "bf16", "target_size": 768}},
                    "generation": {"images": [], "qas": [["q1", None]], "max_gen_len": 8192, "temperature": 1.0},
                },
                1: {"inference_solver": "ChameleonXLLMXForConditionalGeneration"},
            },
            identifiers=["model.embed_tokens.weight"],
            file_256=[
                "6b71408a7c574d98f00114ab770ac6addc71471770456e482e7b5ec641c02345",
                "1d5d8d5532bae0f32ba35d10d411e506d61e4378dc9fc338f2b1e6af2aa322ec",  # 768
                "a8fe636bbee30fef06dcd8e806ffc65b2aed0ad08a07fdc62f35717d0f851be5",  # 512 multi
                "6420fa13483576d46263996627ba7add2237a01f46dedd3b7750112c0cc2d95b",  # 512
            ],
            layer_b3=["6cd6b3caaea270feb5aff8e9fec205a27da4f48a1e740e63dc9a08f16e70a656"],
            layer_256=["eaa882db6a69cf8ed0104a15b2cdbbb570a23a06ab8c8f65f4c6c21719c6ba25"],
        ),
    )
    repo = "openai/clip-vit-large-patch14"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vit",
            series=series,
            comp=comp,
            repo=repo,
            pkg={0: {"transformers": "CLIPTextModel"}},
            identifiers=["text_model.encoder.layers.0.mlp.fc1.weight", "clip-l"],
            file_256=[
                "cb0cba1ead482a850532ebe5ff6b5c8d4456aee32a5228acf0a31e7d9472415e",  # long vit best
                "39e79c916feca4ddf546d9fe923e664714b59ea61074f7228037d17c302f3d17",  # vit l detail improved hit gmp
                "893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082",  # flux/shuttle 3 aes
                "778d02eb9e707c3fbaae0b67b79ea0d1399b52e624fb634f2f19375ae7c047c3",  # playground 2.5
                "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",  # playground 2.5 fp16
                "71e183d11db0c6b6282a4d9e0abb74125edc8692393e89ed8ee5571005f35cb1",  # sd3.5 fp16
                "5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc",  # sdxl diffusers
                "87c1c0b0894c9e9e10b962e597e8d64dd3a3a2d372c389922b335a53c250b2ae",  # L
                "bd289dd57fee86bc8816b55919a2b03f9c3c75af6025e21777325a6730872325",  # jaguar mlx
                "8377b1ca9d88fe06ec483dd7b3cfc62e5e8dbf8ddd252f455e79d659fa0553c5",  # ssd-1b
                "5487ea0eee9c9a9bff8abd097908d4deff3ae1fa87b3b67397f8b9538139d447",  # ssd-1b fp16
                "92b998a9a64549bfa05c019bde114be6681549a0c79caee903fe30c9444d08b9",  # vega
                "1e090d6a828fd92401be5f83e615fd7b4fb1f4a22e9af9040a38f602e839317c",  # vega fp16
                "11807cb2522cfe99240e5ee2bbeb1ccb42cecca2215102ee872567c7773b28b9",  # flux
                "d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334",  # sd1
                "77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e",  # sd1 fp16
                "b70c11ad5d7e9abf6109348908f599ea382f8019e1f36910bbc8ebecde936633",  # hidream i1
                "fc42badf529dd83f2f7c3d20fe6bda1e22036162f37c4c668b9e130884e20561",
                "e27bafa0b3029ad637ef3ace24ce1efe85b8d0dbd22e03a2e70bda6fc88963a1",  # onnx
            ],
            layer_b3=[
                "f58a22a381f79985b6d38782f6110a52c2f319b40fdedd3b88b24945dfcbdf64",
                "8faa00b8fd1dbd9286a7237df18caeb8c91af100a6813849b6bae272a01dd7b7",
                "ab5bebc98299c155251a06deccde599ba0128038ee3ce021e8c59a45f58f72c0",
                "c70e9d86a9dcbbbe7c269ef9dfac96ce9c96c46922577338cc1902e5fe936315",
                "f285e9b7b70745df81adc8b558ec74b536b79b6fc02a453ecc61ea9d13f25f1a",
                "7ab17bfa06ab8d65840997ef641f3f593d096860e20141f1eeb0169d131c1c23",
                "2737d3f327e8176dbb549b9c5c4994821430a6c3b07e3bbc925d97511c802636",  # jaguar mlx q8
                "58a826a4a5fe555b4df188a1ebc0d8d9c96cedae3a26ce84c247861dbb93388f",  # sd1
                "1540fd8844898960e18ce8fd153e5f21a8c446bd8c4d6f536a7cf11418f02bf3",  # sd1
                "c4c9caccdbec12b965d93688c521893f75e0bf9a5e0aad70a6a962b669e7b9d5",  # vega
                "e43fae8d5fd1e562607da172369cc0c5ec99b834e42502e682287ff7d12baacc",  # vega fp16
                "c6f79f7416a882891957b815fbdfd6edfaa253c43970b1a25ef14e217599c7bc",  # flux
                "daf5e09f67ad09a909f58a01298fec0132324634cb8fca2a604c3a240c2c453f",  # jaguar mlx q8
                "3f62bfb6bbde05f01435129326166c44aeb113ac0d9f735f31ed3f7dd04f6980",  # hidream i1
                "22f866f3c96a92bc61e9965cf366d706db942ad047ba8cb82109edcd4e68fa40",  # sd3 turbo
                "f3fa9d7a8f15741621c1fe82f8a1bcc5c601c900d947ac09fba7016615a252a5",  # shap-e
            ],
            layer_256=[
                "48daa3d8f939972e69f044533a4312a941971c18c78255f5e555fa26faf664c1",
                "60f5734a74c342be8b0011fc704e718431839790bcfdc7d7004fc39d70f7fec6",
                "6e76e25b4a55dddfa2eecf4b7ab189a8148658a9f6df165c00170f6ce661033c",
                "2d5249df489fec9137cc3a5e9bda499dd9b72a957ddd8e7ad4e99ff3684bad99",
                "3bf085e701713ed3e79775dafea375c3e2a43659ad1ee788b1b393c0aeff9f0e",
                "efb7976800692772e449c81a739339f59394886590ff3f768b0f9ddd87d2a94c",
                "9b0ac8d127c6c457b2eb8c7236f18c4e4ba9e8bbf27130aa8fe854d7c3f7b1e0",
                "24a9ee3d60cdde6c967f08e4b2ec7088fe1bfe308c6896e73caa874860570a5c",
                "5d6d9d0cc7943eb1b8c16862bfd5bee5c3766d0df027ec837e90fac715ac2bd3",
                "68fb122f7d6c3cfbef320341b2af8f5916678e36a69ed36fa8cfcb19e7d5c43d",
                "11807cb2522cfe99240e5ee2bbeb1ccb42cecca2215102ee872567c7773b28b9",
                "50c46cdddbe9f0162278c69b9a1f818519330e3a91b994272e19b5c789670471",  # jaguar mlx q8
                "ffe1c4f55e07c2010ace7b9cf35798bb9f431bc954a32784e5acbdc16acc0364",  # hidream i1
                "146ea48d234e05a934db9d8988e9a9dd86b2ac70f535eaa550ecb0ee23ec135e",  # sd3 turbo
                "d97560cf9704cf71711f6121df2bf55e55a1eda4b574a6ddba074767420bc8c3",
            ],
        )
    )
    repo = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vit",
            series=series,
            comp=comp,
            repo=repo,
            pkg={0: {"transformers": "CLIPTextModelWithProjection"}},
            identifiers=["31.self_attn.k_proj.weight", "text_model.encoder.layers.22.mlp.fc1.weight", "clip-g"],
            file_256=[
                "ca18e0c67c1ef1e64cac22926266765b60688f692307ecc06283d987c5768134",  # seaart furry g
                "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",  # modelspec sai
                "fa5b2e6f4c2efc2d82e4b8312faec1a5540eabfc6415126c9a05c8436a530ef4",  # playground 2.5
                "b84f413eebecbd049b72874c1df533a516510cb5a2489ae58c7e320209cf0ebe",  # ssd1b
                "d3df577f6e3799c8e1bd9b40e30133710e02e8e25d0ce48cdcc790e7dfe12d6d",  # ssd1b fp16
                "943a2924ee888295a156dd47089d67181d633b782337890af11ef4b15af17ec5",  # vega
                "5b98e4a57a9292eeb819d67e2d2100f66f17db723cde4ecea27a7c3741160d0c",  # vega fp16
                "4d6effa7a5e600cabf7528ed7234146a13ead1b2c151211d706b293a060b112a",  # hidream i1
                "3a6032f63d37ae02bbc74ccd6a27440578cd71701f96532229d0154f55a8d3ff",  # modelspec sai
                "162042ac6556e73f93d4172d4c67532c1cbe4dc7a6a8fa7e44dd2e3d7cbb772b",  # onnx
            ],
            layer_b3=[
                "d754db276f2d89d2808abb7086b3b8eccee43ac521c128d21a071f3a631474a8",
                "2eb93685b34719e1d1e0541d8902b0a592d95848f80657e32816cf3b152a0f31",
                "e253a5cf3a6242c58037abd6b378bf0281f278e441f28dff7ca1bcfcd3cd6bd8",  # ssd1b
                "16d0eec4e55b0aa63cdca4e4d36f78f66a4b1b9605ce3b1089305026f853c3d2",  # ssd1b fp16
                "f606463295ecf3bae8920d3d45bb9d180793418b3d08c3e84d4c4135c7dc2aa5",  # vega
                "7060993a5eb32d94d1ea8aef7a7301e7be73b199c639c63f8f7cfbfcd2abf10e",  # vega fp16
                "b92af95334c657371af6051a91374a41b5455907fa6622bb66a8c112dc511600",  # hidream i1
            ],
            layer_256=[
                "270e998633eb22145100a3889a62ca270d5080654735e5ff8dda09a7c233af8d",
                "df18800c2a9d9318c4323d991a0fb24a6a9afceb41bea203812f60517c301536",
                "4c228b104f6b9b383e0808c9baa1998957f5125d8f90a4d98c1a86e71edd72dc",  # ssd1b
                "f7fc81d8b5ae91ec28a5106ecc0d067be9a94fd3f394c4aa4686ed131ce5a5b3",  # ssd1b fp16
                "61ab42bd5c0fcb9fd3db1d4014cb844ccae8dc17fd69a108cf077a573d092946",  # vega
                "6c64e36cdda3bec7067e94b05619f882f5d31070792acaadac60ddbef580453a",  # vega fp16
                "43c9e64995b485a7f128771c48defce128640df28e65c7f79537d472f43ebe46",  # hidream i1
            ],
        )
    )
    repo = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vit",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {"transformers": "CLIPModel"},
            },
            file_256=[
                "036e6e2bd49697511f4f8b8cb5ee465f93025f7a69a145eadeb9a881ace9b18d",
                "0084e75319a50ad85ef45377bad5bc38f2f58824459eb690048d51c9f8863be5",  # open clip
                "64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161",  # wan sai
            ],
            layer_b3=[
                "227f26ed63120b9034f4a0c90b6b37eede721a8260f2c1e8f7ea3ccc0d109e7e",
                "3a38ffd1b60499cf2f451f3065079ff26efb9190a86f23ad1c8d993bbeb9af05",  # open clip
                "ce06cf1fd684269ee96631b2bf9334c6ecde6a84a55760dfa0d9d2a6411f28e4",  # wan sai
            ],
            layer_256=[
                "130a94ed12569e099196a6ca27388181922e20148dee5bcb58c5e309acfc2352",
                "cfdbd3fd2b90b64ba12d395a62dd7c3c3ea3e811f0a54593e91bae6516ca5061",  # open clip
                "9125ce5970c649d6f9368c25493d3aaa6b41e224d4cc427e955115f7b7e53d1c",  # wan sai
            ],
        )
    )
    repo = "zai-org/chatglm3-6b"  # formerly THUDM
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="aet",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {"transformers": "AutoModel"},
            },
            file_256=[
                "0054d03310248928fdabdeef3fdc753170218dc49a1e9eb5f98323e27683f654",  # kolors
                "b1052386eac358a18add3d0f92521c85ab338979da8eeb08a6499555b857f80d",
            ],
            layer_b3=[
                "a45dfba6a9fa8739777c76deb845fc9589b40f88670d3ce4661646a7b7b1d481",  # kolors
            ],
            layer_256=[
                "174924fd7a07f370bb6fcd1ad07a73eecb7de901f15eefb80f420c1042c47d44",  # kolors
            ],
        )
    )
    base_arch, base_series, base_comp = tag_base_model(repo_path="Qwen/Qwen2-7B-beta", class_name="Qwen2Model")
    repo = "ByteDance-Seed/BAGEL-7B-MoT"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=base_series,
            comp=series,
            repo=repo,
            pkg={0: {"Bagel": "app"}},
        )
    )


def add_mir_audio(mir_db: MIRDatabase):
    """Create MIR audio modality entries"""
    repo = "facebook/audiogen-medium"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "audiocraft": "models.AudioGen",
                    "generation": {"duration": 5},
                    "stage_2": {
                        "audiocraft": ".data.audioaudio_write",
                        "generation": {"strategy": "loudness", "loudness_compressor": True},
                    },
                }
            },
        )
    )
    repo = "parler-tts/parler-tts-tiny-v1"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "parler_tts": "ParlerTTSForConditionalGeneration",
                    "generation": {"return_tensors": "pt"},
                },
            },
        )
    )
    repo = "Zuellni/snac-24khz-ST"
    series, comp = tag_model_from_repo(repo)
    (
        mir_db.add(
            mir_entry(
                domain="info",
                arch="gan",
                series=series,
                comp=comp,
                repo=repo,
                pkg={
                    0: {
                        "snac": "SNAC",
                    },
                    "1": {
                        "mlx_audio": "tts.generate.generate_audio",
                    },
                },
                file_256=["e61ae2f638f56ee07a37592cd5a6a9e7d642560ddc78a76ee4a7f96d6922f1be", "973ee1be4032319fd9685ec54eee1b93e79c7bc98c786e67f17c04669714f11d"],
                layer_b3=["18307b00460a64cc4893f9061592ce8d7e15b70fc54065cc8ae0f0155381ec46", "d599b1bb36dee3cee4674b7922fcd69e5ec05b74413f611d21cfdfdf8f9b6119"],
                layer_256=["35ba9aa1feb931010559a178fcac243673d2efdd1396a4b69d406c9853a88300", "5a22c4707ed6c928043f23b59f2d102a579db3a9af41cf6e60d7c3958f182841"],
            )
        ),
    )
    repo = "parler-tts/parler-tts-large-v1"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "parler_tts": "ParlerTTSForConditionalGeneration",
                    "generation": {"return_tensors": "pt"},
                },
            },
        )
    )
    repo = "hexgrad/Kokoro-82M"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="gan",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {"kokoro": "KPipeline"},
                1: {
                    "mlx_audio": "tts.generate.generate_audio",
                    "generation": {"audio_format": "wav", "join_audio": True, "verbose": False},
                },
            },
            file_256=[
                "5a5cb3d87478f2e74dfca208ee52209ccfce024095e137097fd276026506e45f",
                "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
            ],
            layer_b3=[
                "3e9b5017cfe67a7804ac717b18b6add42ffc0bd3353490df2bcc520eaaef79b6",
                "379660a87a64524bab69a267e3d9580f04b5eec4f7e3fbd48c6597d164d9b17d",  # safetensors
                "997f154f5a78879ef3ba1a1556977c40b28b9c21076b8f583f752c57ecc36e93"  # pytorch
                "2dc3dba29452b85ea85266084a6248f9e0efe642d5f75b43e64f25b9f2837f92",
            ],
            layer_256=[
                "dbedf0e2115aa309b92689f86534be4a77b91d7900365e1717879fbb19b849f6",
                "2c68574571b3f9229e015a909788116ea2251142e29c1bd5c687863192124e8b",
            ],
        )
    )
    repo = "freddyaboulton/silero-vad"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="stst",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "onnx": "onnx",
                },
                1: {
                    "mlx_audio": "tts.generate.generate_audio",
                    "generation": {"audio_format": "wav", "join_audio": True, "verbose": False},
                },
            },
            file_256=["591f853590d11ddde2f2a54f9e7ccecb2533a8af7716330e8adfa6f3849787a9"],
            layer_b3=[
                "41ca5931452b3ffee588c6c7e5bd327c4e914141604eaf3fd05f4a790ac83bb2",
                "7dc736cd5d840182792bde4edfbf5ddc5aeaf16826a9c72d1ba8166c1e3fab9b",
                "6e2c1bdbad74f56663ffb5710c7cb849a2b91ba331d81acdba47a21f69107434",  # onnx
                "ab5ff443aece9171af5e7603d0b4309d3ecc934e3940ccedefff10f0b54b931e",  # onnx vad
                # "7939427700c3b4d91428a490bde1a6d893f63ee5d79b86f68de9e89c7094d3e7"  # onnx # <- clip-g ?? unet? inaccurate test at layer level
            ],
            layer_256=[
                "2ffef1834d5fe14ad8db58fc78d769d5dc38dda5eddbfc396786f74b326215fd",
                # "94ea015f5f7f65b1d8e80f7d52859535e7761d7ed2752e24d57a8d9d9da96672", # onnx lose reliability with layer search apparently
            ],
        ),
    )
    repo = "facebook/wav2vec2-conformer-rope-large-960h-ft"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="stst",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "transformers": "Wav2Vec2ConformerForCTC",
                },
            },
            file_256=["97bb9761fb71ec1225100bc81ccf7d002e0d0ba3d0604c1fd2dbda7d7d491f1d"],
            layer_b3=["6c9c5642aa8dce62bcb3eb577bc519619a2d868005c767c5e65371c583a8a8eb"],
            layer_256=["1afcfda68307a75caa1a1c4456cf97e20c7914e8aba828006e9fe17e8675a79d"],
        ),
    )
    repo = "canopylabs/orpheus-3b-0.1-ft"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "orpheus_tts": "OrpheusModel",
                    "generation": {"max_model_len": 2048},
                },
                1: {
                    "mlx_audio": "tts.generate.generate_audio",
                    "generation": {"audio_format": "wav", "join_audio": True, "verbose": False},
                },
            },
        )
    )
    repo = "OuteAI/OuteTTS-0.3-1B"
    series, comp = tag_model_from_repo(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {"outetts": "InterfaceHF"},
                1: {
                    "mlx_audio": "tts.generate.generate_audio",
                    "generation": {"audio_format": "wav", "join_audio": True, "verbose": False},
                },
            },
        )
    )


def add_mir_lora(mir_db: MIRDatabase):
    """Create MIR lora entries"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="dmd",
            comp=sdxl_series,
            repo="tianweiy/DMD2",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {}},
                    "generation": {"num_inference_steps": 4, "guidance_scale": 0, "timesteps": [999, 749, 499, 249]},
                    "scheduler": {"ops.scheduler.lcm": ""},
                }
            },
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
            comp=sdxl_series,
            repo="radames/sdxl-DPO-LoRA",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {}},
                    "generation": {"guidance_scale": 7.5, "num_inference_steps": 4},
                    "scheduler": {"ops.scheduler.dpm": {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True, "order": 2}},
                },
            },
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
            comp=sdxl_series,
            repo="jasperai/flash-sdxl",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {}},
                    "scheduler": "ops.scheduler.lcm",
                }
            },
            file_256=["afe2ca6e27c4c6087f50ef42772c45d7b0efbc471b76e422492403f9cae724d7"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp="pixart-alpha",
            repo="jasperai/flash-pixart",
            pkg={
                0: {"diffusers": {"load_lora_weights": {}}},
            },
            file_256=["99ef037fe3c1fb6d6bbefdbb85ad60df434fcc0577d34c768d752d60cf69681b"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp=sd3_series,
            repo="jasperai/flash-sd3",
            pkg={
                0: {"diffusers": {"load_lora_weights": {}}},
            },
            file_256=["85fce13c36e3739aa42930f745eb9fceb6c53d53fb17e2a687e3234c1a58ee15"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp=sd1_series,
            repo="jasperai/flash-sd",
            pkg={
                0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 4, "guidance_scale": 0}},
            },
            file_256=["99353444c1a0f40719a1b3037049dbd24800317979a73c312025c05af3574a5f"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            comp=sdxl_series,
            repo="ByteDance/Hyper-SD",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 1.0}}}},
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
            comp=dev_series,
            repo="ByteDance/Hyper-SD",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 0.125}}}},
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
            comp=sd3_series,
            repo="ByteDance/Hyper-SD",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 0.125}}}},
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
            comp=sd1_series,
            repo="ByteDance/Hyper-SD",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
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
            comp=sdxl_series,
            repo="latent-consistency/lcm-lora-sdxl",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {"fuse": 1.0}},
                    "scheduler": {"ops.scheduler.lcm": {"timestep_spacing": "trailing"}},
                    "generation": {"num_inference_steps": 8},
                },
            },
            file_256=["a764e6859b6e04047cd761c08ff0cee96413a8e004c9f07707530cd776b19141"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp=ssd_series,
            repo="latent-consistency/lcm-lora-ssd-1b",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 8}}},
            file_256=["7adaaa69db6f011058a19fd1d5315fdf19ef79fcd513cdab30e173833fd5c59b"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp=vega_series,
            repo="segmind/Segmind-VegaRT",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "gen_kwargs": {"num_inference_steps": 8}}},
            file_256=["9b6e8cd833fa205eaeeed391ca623a6f2546e447470bd1c5dcce3fa8d2f26afb"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp=sd1_series,
            repo="latent-consistency/lcm-lora-sdv1-5",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 8}}},
            file_256=["8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lightning",
            comp=sdxl_series,
            repo="ByteDance/SDXL-Lightning",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 4, "guidance_scale": 0}}},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="pcm",
            comp=sdxl_series,
            repo="wangfuyun/PCM_Weights",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
            file_256={
                "0365f6107250a4fed1b83e8ae6a070065e026a2ba54bff65f55a50284232bbe6": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "04ea827435d5750e63d113dc509174b4f6e8a069ff8f91970c3d25299c10b1f8": {"num_inference_steps": 16},
                "7eb353b2abcaabab6251ba4e17d6cbe2e763feb0674b0f950555552212b44621": {"num_inference_steps": 16},
                "a85cf70ac16ed42011630a5cd6b5927722cb7c40a2107eff85e2670f9a38c893": {"num_inference_steps": 4},  # float16
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
            comp=sd1_series,
            repo="wangfuyun/PCM_Weights",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
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
            comp=sd3_series,
            repo="wangfuyun/PCM_Weights",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
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
            comp=sdxl_series,
            repo="alimama-creative/slam-lora-sdxl",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {}},
                    "scheduler": {"ops.scheduler.lcm": {"timestep_spacing": "trailing"}},
                    "generation": {"num_inference_steps": 4, "guidance_scale": 1},
                }
            },
            file_256=["22569a946b0db645aa3b8eb782c674c8e726a7cc0d655887c21fecf6dfe6ad91"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="slam",
            comp=sd1_series,
            repo="alimama-creative/slam-sd1.5",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp=sdxl_series,
            repo="SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"guidance_scale": 5.0}}},
            file_256=["0b9896f30d29daa5eedcfc9e7ad03304df6efc5114508f6ca9c328c0b4f057df"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp=sd1_series,
            repo="SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep_LoRA",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"guidance_scale": 7.5}}},
            file_256=["1be130c5be2de0beacadd3bf0bafe3bedd7e7a380729932a1e369fb29efa86f4"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="tcd",
            comp=sdxl_series,
            repo="h1t/TCD-SDXL-LoRA",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {}},
                    "generation": {"num_inference_steps": 4, "guidance_scale": 0, "eta": 0.3},
                    "scheduler": {"ops.scheduler.tcd": {}},
                }
            },
            file_256=["2c777bc60abf41d3eb0fe405d23d73c280a020eea5adf97a82a141592c33feba"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="tcd",
            comp=sd1_series,
            repo="h1t/TCD-SD15-LoRA",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
            file_256=["eaecb24a1cda4411eab67275b1d991071216ac93693e8fa0c9226c9df0386232"],
            layer_b3=["90158259812a89beb8874216009c799f420334aac49bbf4fa1bf0ebf4bbd256b"],
            layer_256=["e9825b81bca684126ac3cc8867d2ebc655f74268bc26bea4e4b7e58a52ad6c75"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp=sdxl_series,
            file_256=["a599c42a9f4f7494c7f410dbc0fd432cf0242720509e9d52fa41aac7a88d1b69"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp=dev_series,
            repo="alimama-creative/FLUX.1-Turbo-Alpha",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {"fuse": 0.125}},
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 8, "max_sequence_length": 512},
                }
            },
            file_256=["77f7523a5e9c3da6cfc730c6b07461129fa52997ea06168e9ed5312228aa0bff"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp=sd3_series,
            repo="tensorart/stable-diffusion-3.5-medium-turbo",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 1.0}}, "scheduler": {"ops.scheduler.flow-match": {"shift": 5}}}},
            file_256={"bdcbdfa3ec8ed838b77b1020eea3bc7917a2d42573688a034feb921fde8b1858": {"num_inference_steps": "4"}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp=sd3_series,
            repo="tensorart/stable-diffusion-3.5-large-TurboX",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 1.0}}, "scheduler": {"ops.scheduler.flow-match": {"shift": 5}}}},
            file_256={"fae59d1b749c0d14a8fd4c68cc94eaac92876cee7b91fa75cf8fde3160e09548": {"num_inference_steps": "8"}},
        )
    )


def add_mir_vae(mir_db: MIRDatabase):
    """Create MIR VAE missing from the database"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="tae",
            comp=sd3_series,
            repo="madebyollin/taesd3",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["6f79c1397cb9ce1dac363722dbe70147aee0ccca75e28338f8482fe515891399"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="tae",
            comp=sdxl_series,
            repo="madebyollin/taesdxl",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["ff4824aca94dd6111e0340fa749347fb74101060d9712cb5ef1ca8f1cf17502f"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="tae",
            comp=sd1_series,
            repo="madebyollin/taesd",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="tae",
            comp=dev_series,
            repo="madebyollin/taef1",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["927f7de7f11bbd3b2d5ce402e608d97a7649e0921a9601995b044e8efc81e449"],
        )
    )
    series, comp = tag_model_from_repo("Qwen/Qwen-Image")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLQwenImage"},
            },
            file_256=[
                "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344",  # diffusers
            ],
            layer_b3=[
                "64af8fb08d2054c81ad2aef94965be8fb1366fcc6136cb9222ae046550af014b",  # diffusers
            ],
            layer_256=[
                "42f255440ef1d379a8a731456bc44312a73a8568716caa6100803990cd5ea7dc",  # diffusers
            ],
        )
    )
    series, comp = tag_model_from_repo("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    sr_series_text2v, _ = tag_model_from_repo("Skywork/SkyReels-V2-T2V-14B-720P-Diffusers")
    sr_series_image2v, _ = tag_model_from_repo("Skywork/SkyReels-V2-I2V-14B-720P-Diffusers")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="wan",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {
                    "diffusers": "AutoencoderKLWan",
                    "precision": "ops.precision.float.F32",
                }
            },
            file_256=[
                "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793",  # diffusers
                "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",  # sai
            ],
            layer_b3=[
                "f867543d636029ebfc05b8075e572be0b313a83b0470e56bcf4bbad07a6db010",  # diffusers
                "6b5b229727a2d4e37993687c62c94ff8519a371ab4103c699ff1f5969ca0b433",  # sai
            ],
            layer_256=[
                "121b3974b39263dcca9d644d1b5c9b9251a911b6a8a8e307fcb21ca778e78ed2",
                "364be43a8959012d798d3f98e17d8b5c4b99ba1e70077008dd19acca3ced395e",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="wan",
            comp=sr_series_text2v,
            # no repo here, may conflict
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="wan",
            comp=sr_series_image2v,
            # no repo here, may conflict
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = tag_model_from_repo("Lightricks/LTX-Video")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLLTXVideo"},
            },
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = tag_model_from_repo("rhymes-ai/Allegro")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLAllegro"},
            },
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = tag_model_from_repo("zai-org/CogVideoX-5b-I2V")
    series_fun, _ = tag_model_from_repo("alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose")
    series_wish, _ = tag_model_from_repo("BestWishYsh/ConsisID-preview")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="cogvideox",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLCogVideoX"},
            },
            file_256=["a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e"],
            layer_b3=["246addb8dc798240638bffee4546a3c5c83572139b4a2a602d68b4c4146226eb"],
            layer_256=["43c7e9cb4364e55fd563817f01484ede8a09ff19a8e69eb61a32a12f93d6f66e"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="cogvideox",
            comp=series_fun,
            # no repo here, may conflict
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="cogvideox",
            comp=series_wish,
            # no repo here, may conflict
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = tag_model_from_repo("nvidia/Cosmos-1.0-Diffusion-7B-Video2World")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLCosmos"},
            },
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = tag_model_from_repo("alibaba-pai/EasyAnimateV5.1-7b-zh-diffusers")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLMagvit"},
            },
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = tag_model_from_repo("hunyuanvideo-community/HunyuanVideo-I2V")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLHunyuanVideo"},
            },
            file_256=[
                "95d1fc707c1421ccd88ea542838ab4c5d45a5babb48205bac9ce0985525f9818",  # pt,
                "7c68a6295f9034a88225fbafb1f3258291a08d57a1fdb938233fa57b1b8f4883",
                "fbe5ea338431bc8ba20f7019b474e83379fe5763abfd562adcc04b1c0d35c728",
                "019973c147e0c3462629d8d06bdbdbb83408f3ebd4ea4b4ae21a99c3cdcb54c0",
            ],
            # layer_b3=[],
            # layer_256=[],
        )
    )
    series, comp = tag_model_from_repo("genmo/mochi-1-preview")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLMochi"},
            },
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = tag_model_from_repo("rhymes-ai/Allegro")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {
                    "diffusers": "AutoencoderKLAllegro",
                },
            },
            file_256=["47871a698b18f92f15019d361a81cbc8af4676f8eef9a47fd2b95354a39f831a"],
            layer_b3=["93654cbab7541504d2377c66e72943c7fd9947fca2eb1be01bcc8877c322c1e0"],
            layer_256=["bfd496586118165a13243997101fc7cdd4f855b2d8a73ee2b771a4484c4c2f9f"],
        )
    )
    series, comp = tag_model_from_repo("cvssp/audioldm-s-full-v2")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {
                    "diffusers": "AutoencoderKL",
                },
            },
            file_256=["42f64f7565b23eabde68c9694e39f18b8bba5f7a14f477e7ed4b51e0ea7de8a5"],
            layer_b3=["00959677dae940b9cfdbe5380c8cbb5a6b4951864cd26f8211d74a3d22b4f3de"],
            layer_256=["54d075953d5253a3abac651de070736c1d5510b857a8ab24c624304f428146b6"],
        )
    )

    series, comp = tag_model_from_repo("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="dc",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderDC"},
            },
            file_256=["15a4b09e56d95b768a0ec9da50b702e21d920333fc9b3480d66bb5c7fad9d87f"],
            layer_b3=["cf4ecc6697d18b0663e4eac58203f1dd6d9fb689cf99adfeadbc0019de0c73d0"],
            layer_256=["abfc39d1a6d71f03dde7bc40fec4a90478a97d17ae1688be9aad00e0512b9bde"],
        )
    )
    series, comp = tag_model_from_repo("stabilityai/stable-audio-open-1.0")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="oobleck",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderOobleck"},
            },
            # file_256=[],
            # layer_b3=[],
            # layer_256=[],
        )
    )
    series, comp = tag_model_from_repo("stable-video-diffusion-img2vid-xt")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLTemporalDecoder"},
            },
            # file_256=[],
            # layer_b3=[],
            # layer_256=[],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=sdxl_series,
            repo="madebyollin/sdxl-vae-fp16-fix",
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
            file_256=[
                "235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1",  # modelspec sai
                "1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c",  # diffusers
                "78f6189c8492013e3cac81637a1f657f790a237387f8a9dfd6bfa5fee28eb646",  # ssd1b diffusers
                "6353737672c94b96174cb590f711eac6edf2fcce5b6e91aa9d73c5adc589ee48",  # ssd1b diffusers fp16
                "bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68",  # kolors fp16
                "1598f3d24932bcfe6634e8b618ea1e30ab1d57f5aad13a6d2de446d2199f2341",  # vega / lumina next sft d / auraflow
                "703abdcd7c389316b5128faa9b750a530ea1680b453170b27afebac5e4db30c4",  # pixart a
                "98a14dc6fe8d71c83576f135a87c61a16561c9c080abba418d2cc976ee034f88",  # hyd 1.1
            ],
            layer_b3=[
                "bd5b356b509814025a9cf692710b87116d4fcd0e30a8232ed1db133e908d0e74",  # modelspec sai
                "9106380403dee83238af63ff1738396d2fdff9f6d78d0d9c1d0bf770ae4294d0",  # diffusers
                # "245070a60a25ca080cb4951220c3fb1503da43829930d5f6f7a6770b491eafe1",
                # "50e65a628b5fe379798d8956e4a4e1d4b105c84b329f088d577f7f28c22abc49",  # diffusers fp16 matches sd1
            ],
            layer_256=[
                "c9399a4cd39a180a0bb2af96a8297b9330541e090c21e83317cebb2f7cc651da",  # modelspec sai
                "2240ae134a3b983abf45200c198f07e3d8068012fbbd2f658bbaa1fd6a0629c0",  # diffusers
                # "35641f65ad7ea600cb931dcab556f7503279f1d8d99eda170fe7976d48502a2a",  # diffusers fp16 matches sd1 (incorrect)
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=sdxl_series + sdxl_comp,
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
            file_256=[
                "235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1",  # modelspec sai
                "27ed3b02e09638568e99d4398c67bc654dde04e6c0db61fb2d21dba630e7058a",  # diffusers
                "eb6516ab7e1104d5d1a174a4d65c57835ae38061531d0a2192103aecfb790cc1",  # diffusers fp16
                "e6bb9ea85bbf7bf6478a7c6d18b71246f22e95d41bcdd80ed40aa212c33cfeff",  # modelspec sai vae 0.9
            ],
            layer_b3=[
                "bd5b356b509814025a9cf692710b87116d4fcd0e30a8232ed1db133e908d0e74",  # modelspec sai
                # "9106380403dee83238af63ff1738396d2fdff9f6d78d0d9c1d0bf770ae4294d0",  # diffusers
                # "245070a60a25ca080cb4951220c3fb1503da43829930d5f6f7a6770b491eafe1",
                # "50e65a628b5fe379798d8956e4a4e1d4b105c84b329f088d577f7f28c22abc49",  # diffusers fp16 matches sd1
            ],
            layer_256=[
                "c9399a4cd39a180a0bb2af96a8297b9330541e090c21e83317cebb2f7cc651da",  # modelspec sai
                "2240ae134a3b983abf45200c198f07e3d8068012fbbd2f658bbaa1fd6a0629c0",  # diffusers
                # "35641f65ad7ea600cb931dcab556f7503279f1d8d99eda170fe7976d48502a2a",  # diffusers fp16 matches sd1 (incorrect)
            ],
        )
    )

    repo = "shuttleai/shuttle-jaguar"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=tag_model_from_repo(repo)[0],
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
            file_256=[
                "6fdfa2add4f04d94f36157cbb0197f97966b612e3f8eff4095315aefea74b904",
            ],  # q8,
            layer_b3=[
                "0ebf9b7010accc44e219e355dd24bf1e3128004093c0c1dfc06f88c0a39fdbdd",
                "d0e7ef3c4af06fa08b4c0485a073e2df55f7b1e9e3ba8f7b261688bc562568f0",  # q8
            ],
            layer_256=[
                "9b28f36873ea283905094a64e1ccb7cfc2b0f0aa166201d0ca63807ac37caa7b",  # q8
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=dev_series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
            file_256=[
                "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",  # dev
                "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",  # dev diffusers
                "8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94",  # lumina2
                "8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc",  # sd3 turbo
            ],
            layer_b3=[
                "b6db93ed78c4a10d69e80831c1b8fbc1447f04e9b3d494889ee2056b98d41f17",  # diffusers
                "a8a3ebdec4d7b38d65b7169d3604c19b587330e5e66f69ebf0ded56a24ec6903",  # lumina2
                # "245070a60a25ca080cb4951220c3fb1503da43829930d5f6f7a6770b491eafe1",
            ],
            layer_256=[
                "7950e4f3897c75affaa5f9f3c51c88b4d9a27bfd9b05ad41c3f71d8c1c620b89",
                "79d2bfe93a2ac037cdc59ccb5576e32d00d75d4741fba49fc7e82b9724928216",  # diffusers
                "8f084dc91fd5b481875bc9c86a4ef05e5f176896b7d31c6a5c2ce45c2e174004",  # dev diffusers
                "322e01bd511e20bc2a3c27cd611f81ed85f0046b7c023b5622c2c9a5b8b34f80",  # lumina2
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="eq",
            comp=sdxl_series,
            repo="KBlueLeaf/EQ-SDXL-VAE",
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="ms-lc-eq",
            comp=sdxl_series,
            repo="Anzhc/MS-LC-EQ-D-VR_VAE",
            pkg={
                0: {
                    "diffusers": "AutoencoderKL",
                },
            },
        )
    )
    repo = "ucsd-reach/musicldm"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=tag_model_from_repo(repo)[0],
            # no repo here, may conflict
            file_256=[
                "16e0c6c7c34e459c19500cc15cf538e6331db14969ea15917caa9b0966e44fd4",
            ],  # q8,
            layer_b3=[
                "c5c32b3fb3e73799838836ccce27d883254254daecd10f86ba8ddc55214014e0",
            ],
            layer_256=[
                "1610c0ce39d1379091eb9ab2a4d14a8567e0f1a5dc6cca40fc0fa6f8e4e97c0f",
            ],
        )
    )

    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=sd1_series,
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
            file_256=[
                "0b204ad0cae549e0a7e298d803d57e36363760dec71c63109c1da3e1147ec520",  # ckpt ema original ema pruned
                "95f26a5ab04779d5467d1fcecaf93160ffa523afe399b835b3e1bb77ff2d937a",  # safetensors ema original ema pruned
                "32db726da04f06c1b6b14c0043ce115cc87a501482945c5add89a40d838fcb46",  # safetensors ema diffusers
                "c6a580b13a5bc05a5e16e4dbb80608ff2ec251a162311590c1f34c013d7f3dab",  # ckpt mse original ema pruned
                "735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75",  # safetensors mse original ema pruned
                "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",  # safetensors mse diffusers
                "a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c",  # safetensors diffusers
                "4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c",  # safetensors fp16 diffusers
            ],
            layer_b3=[
                "82e2dc440a23d78bb91df8c9fce069a8512da51f8f54ea29e3431f545808171e",  # safetensors original
                "2230487833925a104bee96e7ecfebaa4c3c43cc426c7a5b863f2584313dd4833",  # safetensors diffusers
            ],
            layer_256=[
                "e43f3a227b5ecb43a6272fa92ed6011d2e9abcadadd1032dfa7ea7f875f9d5bd",  # safetensors original
                "2494154245becf98891be884f943276aa3f54e9b3f0ea1042903fc15fba488f3",  # safetensors diffusers
            ],
        )
    )
