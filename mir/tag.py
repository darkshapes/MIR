# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import dataclass, field
from typing import Callable

from mir.generate.transformers.raw_data import PrepareData


@dataclass
class MIRTag:
    """Represents a MIR tag associated with a specific domain and model data.\n

    Attributes:\n
        domain: The domain of the MIR tag.
        prepared_data: Object containing prepared model data.
        arch: The architecture component of the MIR tag (generated).
        series: The series component of the MIR tag (generated).
        pkg Package information associated with the MIR tag (generated).
        tokenizer_pkg Dependency package information associated with the MIR tag (generated).
    """

    domain: str
    data: PrepareData
    arch: str = field(init=False)
    series: str = field(init=False)
    pkg: str = field(default_factory=str)
    tokenizer_pkg: str = field(default_factory=str)

    def __post_init__(self) -> None:
        """Initializes MIRTag instance, setting up database connection and generating package and MIR tag information."""
        from mir.maid import MIRDatabase

        self.mir_db = MIRDatabase()
        self.pkg = self.generate_pkg(pkg=self.data.model)
        if hasattr(self.data, "tokenizer") and self.data.tokenizer:
            self.tokenizer_pkg = self.generate_pkg(pkg=self.data.tokenizer)  # type:ignore
        self.generate_arch()
        self.generate_series_and_comp(repo_title=self.data.repo_path)

    def generate_pkg(self, pkg: Callable) -> str:
        """Generates package information for the MIR tag based on class.
        :param pkg: A class object (model, tokenizer, etc) to build a tag from"""

        return f"{pkg.__module__}.{pkg.__name__}"

    def generate_arch(self) -> None:
        """Generates the architecture part of the MIR tag based on prepared data.\n
        :raises ValueError: If no suitable tag can be determined."""
        from mir.generate.from_module import to_domain_tag

        library = self.pkg.split(".")[0]
        arch = to_domain_tag(library, **self.data.config_params)
        if not arch:
            if self.data.model_params:
                if arch := to_domain_tag(library, **self.data.model_params):
                    pass
                raise ValueError(f"Unable to determine MIR prefix from {self}")
            else:
                raise ValueError(
                    f"Unrecognized model type, \
                        no tag matched {self.data.name} \
                            with {self.data.config_params} or {self.data.model_params}",
                )
        self.arch = arch

    def generate_series_and_comp(self, repo_title: str, decoder=False) -> None:
        """Generates the MIR tag components from a repository title.\n
        :param repo_title: The title of the repository from which to derive the MIR tag.
        :param decoder: Boolean flag indicating if the model is a decoder.
        :return: A tuple containing the cleaned tag string and suffix."""

        import re

        from mir import BREAKING, PARAMETERS

        root = "decoder" if decoder else "*"
        repo_title = repo_title.split(":latest")[0]
        repo_title = repo_title.split(":Q")[0]
        repo_title = repo_title.split(r"/")[-1].lower()
        pattern = r"^.*[v]?(\d{1}+\.\d).*"
        match = re.findall(pattern, repo_title)
        if match:
            if next(iter(match)):
                repo_title = repo_title.replace(next(iter(match))[-1], "")
        parts = repo_title.replace(".", "").split("-")
        if len(parts) == 1:
            parts = repo_title.split("_")
        subtraction_prefixes = r"\d.b-|\-rl|tiny|large|mlx|onnx|gguf|medium|base|multimodal|mini|instruct|full|:latest|preview|small|pro|beta|hybrid|plus|dpo|community"

        pattern_2 = re.compile(PARAMETERS)
        clean_parts = [re.sub(pattern_2, "", segment.lower()) for segment in parts]
        cleaned_string = "-".join([x for x in clean_parts if x])
        cleaned_string = re.sub(subtraction_prefixes, "", cleaned_string)
        cleaned_string = re.sub("-it", "", cleaned_string.replace("-bit", "")).replace("--", "-")
        cleaned_string = cleaned_string.replace("-b-", "")
        suffix_match = re.findall(BREAKING, cleaned_string)  # Check for breaking suffixes first
        if suffix_match:
            suffix = next(iter(suffix for suffix in suffix_match[0] if suffix))
            cleaned_string = re.sub(suffix.lower(), "-", cleaned_string).rstrip("-,")
        else:
            suffix = root
        cleaned_string = re.sub(r"[.-]+", "_", cleaned_string.lower()).strip("-_")
        self.series = cleaned_string
        if suffix != "*":
            self.comp = suffix
