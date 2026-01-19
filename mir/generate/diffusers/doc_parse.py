# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import List, Optional, Callable
from pydantic import BaseModel, field_validator
from mir import NFO
from mir.data import PIPE_MARKERS


class DocStringValidator:
    """Handles validation of docstring data and extracted values."""

    @staticmethod
    def normalize_doc_string(doc_string: str) -> str:
        """Normalize docstring by joining lines with spaces.\n
        :param doc_string: Raw docstring text
        :returns: Normalized docstring with newlines replaced by spaces
        """
        return " ".join(doc_string.splitlines())

    @staticmethod
    def is_valid_repo_path(repo_path: Optional[str]) -> bool:
        """Validate that a repository path is valid.\n
        :param repo_path: Repository path to validate
        :returns: True if path is valid (not empty and contains '/'), False otherwise
        """
        return repo_path is not None and repo_path.strip() != "" and "/" in repo_path

    @staticmethod
    def validate_repo_path(repo_path: Optional[str], segment: str) -> Optional[str]:
        """Validate and return repo path, or None if invalid.\n
        :param repo_path: Repository path to validate
        :param segment: Segment text for error reporting
        :returns: Validated repo path or None if invalid
        """
        if not DocStringValidator.is_valid_repo_path(repo_path):
            NFO(f"Warning: Unable to resolve repo path for {segment}")
            return None
        return repo_path


class DocStringParser(BaseModel):
    doc_string: str
    model: Callable
    model_path: str
    pipe_repo: str | None = None
    staged_repo: str | None = None

    @field_validator("doc_string")
    def normalize_doc(cls, docs: str) -> str:
        return DocStringValidator.normalize_doc_string(docs)

    def parse(self) -> dict[str, str] | None:
        candidate, prior_candidate, staged = self.doc_match(PIPE_MARKERS["pipe_variables"])
        if candidate:
            pipe_repo = self._extract_class_and_repo(
                segment=candidate,
                call_methods=PIPE_MARKERS["call_methods"],
                prior_text=prior_candidate,
            )
            motion_adapter = "motion_adapter" in candidate or "adapter" in candidate
            if motion_adapter and pipe_repo:
                staged, prior_candidate, _ = self.doc_match(PIPE_MARKERS["pipe_variables"][2:])  # skip the adapter statements

            staged_repo = (
                self._extract_class_and_repo(
                    segment=staged,
                    call_methods=PIPE_MARKERS["staged_call_methods"] if not motion_adapter else PIPE_MARKERS["call_methods"],
                    prior_text=prior_candidate,
                )
                if staged
                else None
            )

            self.pipe_repo = pipe_repo
            self.staged_repo = staged_repo

    def doc_match(self, prefix_set: List[str] | None = None):
        if prefix_set is None:
            prefix_set = PIPE_MARKERS["pipe_variables"]
        assert prefix_set is not None
        candidate = None
        staged = None
        prior_candidate = ""
        for prefix in prefix_set:
            candidate = self.doc_string.partition(prefix)[2]
            prior_candidate = self.doc_string.partition(prefix)[0]
            if candidate:
                staged = candidate if any(call_method in candidate for call_method in PIPE_MARKERS["staged_call_methods"]) else None
                break

        return candidate, prior_candidate, staged

    def _extract_class_and_repo(
        self,
        segment: str,
        call_methods: List[str],
        prior_text: str,
    ) -> str | None:
        pipe_repo = None
        for method_name in call_methods:
            if method_name in segment:
                if not (repo_segment := segment.partition(method_name)[2].partition(")")[0]):
                    repo_segment = segment.partition(method_name)[2].partition(")")[0]
                pipe_repo = repo_segment.replace("...", "").partition('",')[0].strip('" ')
                if not DocStringValidator.is_valid_repo_path(pipe_repo):
                    for reference in PIPE_MARKERS["repo_variables"]:
                        if reference in segment:
                            pipe_repo = self._resolve_variable(reference, prior_text)
                            break  # Not empty!! 确保解析的路径不是空的！！
                pipe_repo = DocStringValidator.validate_repo_path(pipe_repo, segment)
                return pipe_repo

        return pipe_repo

    def _resolve_variable(self, reference: str, prior_text: str) -> str | None:
        """Try to find the variable from other lines / 尝试从其他行中找到它（例如，多行定义）"""
        var_name = reference
        search = f"{var_name} ="

        for line in prior_text.splitlines():
            if search in line:
                repo_block = line.partition(search)[2].strip().strip('"').strip("'")
                index = repo_block.find('"')
                repo_id = repo_block[:index] if index != -1 else repo_block
                if repo_id:  # Keep trying if empty"
                    return repo_id

        for line in prior_text.splitlines():
            if var_name in line:
                start_index = line.find(var_name)
                end_index = line.find("=", start_index)
                if end_index != -1:
                    repo_block = line[end_index + 1 :].strip().strip('"').strip("'")
                    index = repo_block.find('"')
                    repo_id = repo_block[:index] if index != -1 else repo_block
                    if repo_id:
                        return repo_id

        NFO(f"Warning: {search} not found in docstring.")
        return None
