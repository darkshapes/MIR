---
language:
- en
library_name: mir
---

# MIR (Machine Intelligence Resource)<br><br>A naming schema for AIGC/ML work.

The MIR classification format seeks to standardize and complete a hyperlinked network of model information, improving accessibility and reproducibility across the AI community.<br>
The work is inspired by:
- [AIR-URN](https://github.com/civitai/civitai/wiki/AIR-%E2%80%90-Uniform-Resource-Names-for-AI) project by [CivitAI](https://civitai.com/)
- [Spandrel](https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel/spandrel/__helpers/registry.py) library's super-resolution registry

Example:

> [!NOTE]
> # mir : model . transformer . clip-l : stable-diffusion-xl


```
 mir : model .    lora      .    hyper    :   flux-1
  ↑      ↑         ↑               ↑            ↑
 [URI]:[Domain].[Architecture].[Series]:[Compatibility]
```

## Definitions:

Like other URI schema, the order of the identifiers roughly indicates their specificity from left (broad) to right (narrow)

### Domains


- `dev`: Varying local neural network layers, in-training, pre-release, items under evaluation, likely in unexpected formats<br>
- `model`: Static local neural network layers. Publicly released machine learning models with an identifier in the database<br>
- `operations`: Varying global neural network attributes, algorithms, optimizations and procedures on models<br>
- `info`:  Static global neural network attributes, metadata with an identifier in the database<br>

### Architecture
Broad and general terms for system architectures.
- `dit`: Diffusion transformer, typically Vision Synthesis
- `unet`: Unet diffusion structure
- `art` : Autoregressive transformer, typically LLMs
- `lora`: Low-Rank Adapter (may work with dit or transformer)
- `vae`: Variational Autoencoder

### Series
Foundational network and technique types.

### Compatibility
Implementation details based on version-breaking changes, configuration inconsistencies, or other conflicting indicators that have practical application.

### Goals
- Standard identification scheme for **ALL** fields of ML-related development
- Simplification of code for model-related logistics
- Rapid retrieval of resources and metadata
- Efficient and reliable compatibility checks
- Organized hyperparameter management

> <details> <summary>Why not use `diffusion`/`sgm`, `ldm`/`text`/hf.co folder-structure/brand or trade name/preprint paper/development house/algorithm</summary>
>
> - The format here isnt finalized, but overlapping resource definitions or complicated categories that are difficult to narrow have been pruned
> - Likewise, definitions that are too specific have also been trimmed
> - HF.CO become inconsistent across folders/files and often the metadata enforcement of many important developments is neglected
> - Development credit often shared, [Paper heredity tree](https://www.connectedpapers.com/search?q=generative%20diffusion), super complicated
> - Algorithms (esp application) are less common knowledge, vague, ~~and I'm too smooth-brain.~~
> - Overall an attempt at impartiality and neutrality with regards to brand/territory origins
> </details>

> <details><summary>Why `unet`, `dit`, `lora` over alternatives</summary>
>
> - UNET/DiT/Transformer are shared enough to be genre-ish but not too narrowly specific
> - Very similar technical process on this level
> - Functional and efficient for random lookups
> - Short to type
> </details>

> <details><summary>Roadmap</summary>
>
> - Decide on `@` or `:` delimiters (like @8cfg for an indistinguishable 8 step lora that requires cfg)
> - crucial spec element, or an optional, MIR app-determined feature?
> - Proof of concept generative model registry
> - Ensure compatability/integration/cross-pollenation with [OECD AI Classifications](https://oecd.ai/en/classification)
> - Ensure compatability/integration/cross-pollenation with [NIST AI 200-1 NIST Trustworthy and Responsible AI](https://www.nist.gov/publications/ai-use-taxonomy-human-centered-approach)
> </details>

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65ff1816871b36bf84fc3c37/NWZideVk_pp_4OzQDl96w.png)

massive thank you to [@silveroxides](https://huggingface.co/silveroxides) for phenomenal work collecting pristine state dicts and related information

#
