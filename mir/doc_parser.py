# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import List, Optional, Tuple

from pydantic import BaseModel, field_validator
from mir.config.console import dbuq, nfo
from mir.config.constants import DocParseData, DocStringParserConstants


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
            nfo(f"Warning: Unable to resolve repo path for {segment}")
            return None
        return repo_path

    @staticmethod
    def validate_pipe_class(pipe_class: Optional[str]) -> bool:
        """Validate that a pipe class name is present.\n
        :param pipe_class: Pipe class name to validate
        :returns: True if class name is valid, False otherwise
        """
        return pipe_class is not None and pipe_class.strip() != ""


class DocStringParser(BaseModel):
    doc_string: str

    @field_validator("doc_string")
    def normalize_doc(cls, docs: str) -> str:
        return DocStringValidator.normalize_doc_string(docs)

    def doc_match(self, prefix_set: List[str] | None = None):
        if prefix_set is None:
            prefix_set = DocStringParserConstants.pipe_prefixes
        candidate = None
        staged = None
        for prefix in prefix_set:
            candidate = self.doc_string.partition(prefix)[2]
            prior_candidate = self.doc_string.partition(prefix)[0]
            if candidate:
                staged = candidate if any(call_type in candidate for call_type in DocStringParserConstants.staged_call_types) else None
                break

        return candidate, prior_candidate, staged

    def parse(self) -> DocParseData:
        candidate, prior_candidate, staged = self.doc_match(DocStringParserConstants.pipe_prefixes)
        if candidate:
            pipe_class, pipe_repo = self._extract_class_and_repo(
                segment=candidate,
                call_types=DocStringParserConstants.call_types,
                prior_text=prior_candidate,
            )
            motion_adapter = "motion_adapter" in candidate or "adapter" in candidate
            if motion_adapter and pipe_repo:
                staged, prior_candidate, _ = self.doc_match(DocStringParserConstants.pipe_prefixes[2:])  # skip the adapter statements
            staged_class, staged_repo = (
                self._extract_class_and_repo(
                    segment=staged,
                    call_types=DocStringParserConstants.staged_call_types if not motion_adapter else DocStringParserConstants.call_types,
                    prior_text=prior_candidate,
                    prior_class=pipe_class,
                )
                if staged
                else (None, None)
            )
            if motion_adapter and pipe_class:
                pipe_class = staged_class
                staged_repo = None
                staged_class = None

            if DocStringValidator.validate_pipe_class(pipe_class):
                dbuq(f"class :{pipe_class}, repo : {pipe_repo}, staged_class: {staged_class}, staged_repo:{staged_repo} \n")
                return DocParseData(pipe_class=pipe_class, pipe_repo=pipe_repo, staged_class=staged_class, staged_repo=staged_repo)

    def _extract_class_and_repo(
        self,
        segment: str,
        call_types: List[str],
        prior_text: str,
        prior_class: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        pipe_class = None
        pipe_repo = None
        for call_type in call_types:
            if call_type in segment:
                pipe_class = segment.partition(call_type)[0].strip().split("= ")[-1]
                if prior_class == pipe_class:
                    pipe_class = prior_text.partition(call_type)[0].strip().split("= ")[-1]
                    repo_segment = segment.partition(call_type)[2].partition(")")[0]
                else:
                    repo_segment = segment.partition(call_type)[2].partition(")")[0]
                pipe_repo = repo_segment.replace("...", "").partition('",')[0].strip('" ')
                if not DocStringValidator.is_valid_repo_path(pipe_repo):
                    for reference in DocStringParserConstants.repo_variables:
                        if reference in segment:
                            pipe_repo = self._resolve_variable(reference, prior_text)
                            break  # Not empty!! 確保解析後的路徑不為空!!
                pipe_repo = DocStringValidator.validate_repo_path(pipe_repo, segment)
                return pipe_class, pipe_repo

        return pipe_class, pipe_repo

    def _resolve_variable(self, reference: str, prior_text: str) -> Optional[str]:
        """Try to find the variable from other lines / 嘗試從其他行中查找（例如多行定義）"""
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

        nfo(f"Warning: {search} not found in docstring.")
        return None


def parse_docs(doc_string: str) -> DocParseData:
    parser = DocStringParser(doc_string=doc_string)
    return parser.parse()
