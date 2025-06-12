<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img_src/mir300_dark.png" width="50%">
  <source media="(prefers-color-scheme: light)" srcset="img_src/mir300_light.png" width="50%">
  <img alt="Futuristic, slanted wireframe block type spelling the word 'ZODIAC' using overlapping and interweaving shapes and triangles for the 'O' and 'A'." src="img_src/zodiac_dark_bg.png">
</picture><br><br>

</div>

---
language:
- en
library_name: mir
license_name: LGPL-3.0-only
---

# MIR (Machine Intelligence Resource)<br><sub>A naming schema for AIGC/ML work.</sub>


The MIR classification format seeks to standardize and complete a hyperlinked network of model information, improving accessibility and reproducibility across the AI community.<br>

This repo is a live development implementation, an example of autogenerating model inference parameters and code with the MIR schema. This is the sister repo to our [🤗HuggingFace MIR project](https://huggingface.co/darkshapes/MIR) which is an archive of model state dict layer information that also uses the MIR schema.

MIR is inspired by:
- [AIR-URN](https://github.com/civitai/civitai/wiki/AIR-%E2%80%90-Uniform-Resource-Names-for-AI) project by [CivitAI](https://civitai.com/)
- [Spandrel](https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel/spandrel/__helpers/registry.py) library's super-resolution registry

> [!NOTE]
> ## Example:
> ## mir : model . transformer . clip-l : stable-diffusion-xl
>
>
> ```
> mir : model .    lora      .    hyper    :   flux-1
>   ↑      ↑         ↑               ↑            ↑
>  [URI]:[Domain].[Architecture].[Series]:[Compatibility]
> ```


[![Python application](https://github.com/darkshapes/MIR/actions/workflows/mir.yml/badge.svg)](https://github.com/darkshapes/MIR/actions/workflows/python-app.yml)<br>
![commits per month](https://img.shields.io/github/commit-activity/m/darkshapes/MIR?color=indigo)<br>
![code size](https://img.shields.io/github/languages/code-size/darkshapes/MIR?color=navy)<br>
![Discord](https://img.shields.io/discord/1266757128249675867?color=black)<br><br>


### [installation instructions link](https://github.com/darkshapes/sdbx/wiki/Develop)<br>

</div>



