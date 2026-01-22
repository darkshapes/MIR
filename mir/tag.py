# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import dataclass, field

from mir.model import ModelAttributes
from mir.package import MIRPackage


@dataclass
class MIRTag:
    """Represents a MIR tag associated with a specific domain and model data.\n

    Attributes:\n
        prepared_data: Object containing prepared model data.
        arch: The architecture component of the MIR tag (generated).
        series: The series component of the MIR tag (generated).
        comp The compatibility component of the MIR tag (generated, optional).
    """

    attributes: ModelAttributes
    package: MIRPackage
    decoder: bool = False
    arch: str = field(init=False)
    series: str = field(init=False)

    def __post_init__(self) -> None:
        """Initializes MIRTag instance, setting up database connection and generating package and MIR tag information."""
        self.generate_arch()
        self.generate_series_and_comp()
        if hasattr(self, "comp"):
            self.flat = f"{self.arch}.{self.series}.{self.comp}"
        else:
            self.flat = f"{self.arch}.{self.series}"

    def generate_arch(self) -> None:
        """Generates the architecture part of the MIR tag based on prepared data.\n
        :raises ValueError: If no suitable tag can be determined."""

        arch = self.tag_architecture()  # type: ignore
        assert arch is not None, f"Unrecognized model type, no tag matched {self.attributes.model_name} with {self.attributes}"
        self.arch = arch

    def generate_series_and_comp(self, base_model_label="*") -> None:
        """Generates the MIR tag components from a repository title.\n
        :param repo_title: The title of the repository from which to derive the MIR tag.
        :param decoder: Boolean flag indicating if the model is a decoder.
        :return: A tuple containing the cleaned tag string and suffix."""

        import re

        from mir import BREAKING, PARAMETERS

        repo_path = self.package.repo.split(":latest")[0]
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
            suffix = "*"
            if isinstance(self.attributes, DiffusersModelAttributes) and self.attributes.model_type == "decoder":
                suffix = "decoder"
        cleaned_string = re.sub(r"[.-]+", "_", cleaned_string.lower()).strip("-_")
        self.series = cleaned_string
        if suffix != "*":
            self.comp = suffix

    def tag_architecture(self, library: str, **kwargs) -> str | None:
        """Set type of MIR prefix depending on model type\n
        :param library: Library source of the original data
        :raises ValueError: Model type not detected
        :return: MIR prefix based on model configuration"""
        from mir.data import NN_FILTER

        flags = NN_FILTER["arch"][library]  # pylint:disable=unsubscriptable-object
        if library == "diffusers":
            for module_type, module_obj in kwargs.items():
                module_name = module_obj.__module__
                library_path = f"{library}.models."
                if library_path in module_name:
                    module_name = module_name.replace(library_path, "").split(".")[0]
                    if mir_prefix := [match for match in flags if module_name in flags[match]]:
                        return mir_prefix[0]
        for mir_prefix, key_match in flags.items():
            if any(kwargs.get(param, None) for param in key_match):
                return mir_prefix
        return None


def tag_scheduler(self, scheduler_name: str) -> tuple[str, str]:
    """Create a mir label from a scheduler operation\n
    :param class_name: Known period-separated prefix and model type
    :return: The assembled mir tag with compatibility pre-separated"""
    import re

    series_name = None
    comp_name = None
    patterns = [r"Schedulers", r"Multistep", r"Solver", r"Discrete", r"Scheduler"]
    for scheduler in patterns:
        compiled = re.compile(scheduler)
        match = re.search(compiled, scheduler_name)
        if match:
            comp_name = match.group()
            comp_name = comp_name.lower()
            break
    for pattern in patterns:
        series_name = re.sub(pattern, "", scheduler_name)
    if not series_name:
        series_name = scheduler_name
    series_name.lower()
    assert series_name is not None, "Expected series tag but got None"
    assert comp_name is not None, "Expected compatibility tag but got None"
    return series_name, comp_name


def tag_tokenizer():
    pass
