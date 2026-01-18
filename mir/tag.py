# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import dataclass, field

from mir.generate.transformers.raw_data import PrepareData
from mir.generate.diffusers.raw_data import DPrepareData


@dataclass
class MIRTag:
    """Represents a MIR tag associated with a specific domain and model data.\n

    Attributes:\n
        prepared_data: Object containing prepared model data.
        arch: The architecture component of the MIR tag (generated).
        series: The series component of the MIR tag (generated).
        comp The compatibility component of the MIR tag (generated, optional).
    """

    raw_data: PrepareData | DPrepareData
    arch: str = field(init=False)
    series: str = field(init=False)
    decoder: bool = False

    def __post_init__(self) -> None:
        """Initializes MIRTag instance, setting up database connection and generating package and MIR tag information."""
        self.generate_arch()
        self.generate_series_and_comp(repo_path=self.raw_data.repo_path)

    def generate_arch(self) -> None:
        """Generates the architecture part of the MIR tag based on prepared data.\n
        :raises ValueError: If no suitable tag can be determined."""
        from mir.generate.from_module import to_domain_tag

        library = self.raw_data.model.__module__.split(".")[0]
        if hasattr(self.raw_data, "config_params"):
            arch = to_domain_tag(library, **self.raw_data.config_params)  # type: ignore
        else:
            arch = None
            self.decoder = "decoder" in [self.raw_data.model_params]
        if not arch:
            if self.raw_data.model_params:
                if arch := to_domain_tag(library, **self.raw_data.model_params):
                    pass
                raise ValueError(f"Unable to determine MIR prefix from {self}")
            else:
                raise ValueError(
                    f"Unrecognized model type, \
                        no tag matched {self.raw_data.name} \
                            with {self.raw_data}",
                )
        self.arch = arch

    def generate_series_and_comp(self, repo_path: str, decoder=decoder) -> None:
        """Generates the MIR tag components from a repository title.\n
        :param repo_title: The title of the repository from which to derive the MIR tag.
        :param decoder: Boolean flag indicating if the model is a decoder.
        :return: A tuple containing the cleaned tag string and suffix."""

        import re

        from mir import BREAKING, PARAMETERS

        root = "decoder" if decoder else "*"
        repo_path = repo_path.split(":latest")[0]
        repo_path = repo_path.split(":Q")[0]
        repo_path = repo_path.split(r"/")[-1].lower()
        pattern = r"^.*[v]?(\d{1}+\.\d).*"
        match = re.findall(pattern, repo_path)
        if match:
            if next(iter(match)):
                repo_path = repo_path.replace(next(iter(match))[-1], "")
        parts = repo_path.replace(".", "").split("-")
        if len(parts) == 1:
            parts = repo_path.split("_")
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

    # def generate_pipe_tag(repo_path: str, class_name: str, model_class_obj: Callable | None = None) -> tuple[str, dict[str, dict[Any, Any]]]:
    #     """Create a pipeline article and generate corresponding information according to the provided repo path and pipeline category\n
    #     :param repo_path (str): Repository path.
    #     :param model_class_obj (str): The model class function
    #     :raises TypeError: If 'repo_path' or 'class_name' are not set.
    #     :return: Tuple: The data structure containing mir_series and mir_comp is used for subsequent processing.
    #     """
    #     import diffusers  # pyright: ignore[reportMissingImports] # pylint:disable=redefined-outer-name

    #     if hasattr(diffusers, class_name):
    #         model_class_obj = getattr(diffusers, class_name)
    #         sub_segments = show_init_fields_for(model_class_obj, "diffusers")

    #         else:
    #             mir_prefix = to_domain_tag(**sub_segments)
    #             if mir_prefix is None and class_name not in ["AutoPipelineForImage2Image", "DiffusionPipeline"]:
    #                 NFO(f"Failed to detect type for {class_name} {list(sub_segments)}\n")
    #             else:
    #                 mir_prefix = "info." + mir_prefix

    #         mir_series, mir_comp = list(tag_model_from_repo(repo_path, decoder))
    #         mir_series = mir_prefix + "." + mir_series
    #         repo_path = migrations(repo_path)
    #         # modalities = add_mode_types(mir_tag=[mir_series, mir_comp])
    #         prefixed_data = {
    #             "repo": repo_path,
    #             "pkg": {0: {"diffusers": class_name}},
    #             # "mode": modalities.get("mode"),
    #         }
    #         return mir_series, {mir_comp: prefixed_data}
