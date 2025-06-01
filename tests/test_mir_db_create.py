### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


def test_mir_creation():
    from nnll.monitor.file import nfo
    from mir.mir import mir_entry
    from pprint import pprint

    entry = mir_entry(domain="info", arch="unet", series="stable-diffusion-xl", comp="base", gen_kwargs={"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent", "safety_checker": False}, pipe_kwargs={"use_safetensors": True})
    entry.update(
        mir_entry(domain="model", arch="unet", series="stable-diffusion-xl", comp="base", file_path="/Users/nyan/Documents/models"),
    )
    entry.update(
        mir_entry(
            domain="ops",
            arch="scheduler",
            series="align-your-steps",
            comp="stable-diffusion-xl",
            num_inference_steps=10,
            timesteps="StableDiffusionXLTimesteps",
            deps_pkg=["diffusers"],
            module_path=["schedulers.scheduling_utils", "AysSchedules"],
        )
    )
    entry.update(
        mir_entry(
            domain="ops",
            arch="patch",
            series="hidiffusion",
            comp="stable-diffusion-xl",
            num_inference_steps=10,
            timesteps="StableDiffusionXLTimesteps",
            deps_pkg='["hidiffusion"]',
            gen_kwargs={"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5},
            module_path=["apply_hidiffusion"],
        )
    )
    pprint(entry)


# eta only works with ddim!!!

if __name__ == "__main__":
    test_mir_creation()
