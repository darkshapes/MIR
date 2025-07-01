### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

from pydantic import BaseModel, field_validator
from typing import List, Optional, Tuple


def parse_docs(doc_string: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    parser = DocParser(doc_string=doc_string)
    result = parser.parse()
    return result.pipe_class, result.repo_path, result.staged_class, result.staged_repo


class DocParseResult(BaseModel):
    pipe_class: Optional[str]
    repo_path: Optional[str]
    staged_class: Optional[str] = None
    staged_repo: Optional[str] = None


class DocParser(BaseModel):
    doc_string: str

    pipe_prefixes: List[str] = [
        ">>> adapter = ",
        ">>> pipe_prior = ",
        ">>> pipe = ",
        ">>> pipeline = ",
        ">>> blip_diffusion_pipe = ",
        ">>> gen_pipe = ",
        ">>> prior_pipe = ",
    ]
    repo_variables: List[str] = ["repo_id", "model_id_or_path", "model_ckpt", "model_id", "repo_base", "repo"]

    call_types: List[str] = [".from_pretrained(", ".from_single_file("]
    staged_call_types: List[str] = [
        ".from_pretrain(",
    ]

    @field_validator("doc_string")
    def normalize_doc(cls, docs: str) -> str:
        return " ".join(docs.splitlines())

    def parse(self) -> DocParseResult:
        pipe_doc = ""
        staged = ""
        prior_candidate = ""
        for i, prefix in enumerate(self.pipe_prefixes):
            candidate = self.doc_string.partition(prefix)[2]
            prior_candidate = self.doc_string
            if candidate:
                if any([x for x in self.staged_call_types if x in candidate]):  # second-to-last
                    staged = candidate
                pipe_doc = candidate
                break

        pipe_class, repo_path = self._extract_class_and_repo(
            segment=pipe_doc,
            call_types=self.call_types,
            prior_text=prior_candidate,
        )
        # print(self.doc_string)
        staged_class, staged_repo = (
            self._extract_class_and_repo(
                segment=staged,
                call_types=self.staged_call_types,
                prior_text=prior_candidate,
            )
            if staged
            else (None, None)
        )
        return DocParseResult(pipe_class=pipe_class, repo_path=repo_path, staged_class=staged_class, staged_repo=staged_repo)

    def _extract_class_and_repo(self, segment: str, call_types: List[str], prior_text: str) -> Tuple[Optional[str], Optional[str]]:
        for call_type in call_types:
            if call_type in segment:
                pipe_class = segment.partition(call_type)[0].strip()
                if "=" in pipe_class:
                    pipe_class = pipe_class.partition("= ")[2]
                repo_part = segment.partition(call_type)[2].partition(")")[0]
                repo_path = repo_part.replace("...", "").partition('",')[0].strip('" ')
                if not repo_path or "/" not in repo_path:
                    for reference in self.repo_variables:
                        if reference in segment:
                            repo_path = self._resolve_variable(reference, prior_text)
                            break  # Not empty!! 確保解析後的路徑不為空!!
                if not repo_path:
                    print(f"Warning: Unable to resolve repo path for {segment}")
                return pipe_class, repo_path

        return None, None

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

        print(f"Warning: {search} not found in docstring.")
        return None

    # def _resolve_variable(self, reference: str, prior_text: str) -> Optional[str]:
    #     var_name = reference
    #     search = f"{var_name} ="

    #     for line in prior_text.splitlines():
    #         if search in line:
    #             return line.partition(search)[2].partition('"')[0].strip('"').strip("'")
    #     print(f"{search}\n\n{prior_text}")
    #     return None
