### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""神经网络的数据注册"""

# pylint: disable=possibly-used-before-assignment, line-too-long
from typing import Any, Callable, List, Optional
import os
import sys

# from nnll.monitor.file import debug_monitor, dbug
# from nnll.monitor.console import nfo
from mir.json_cache import JSONCache, MIR_PATH_NAMED  # pylint:disable=no-name-in-module
# from logging import Logger, INFO

# nfo_obj = Logger(INFO)
# nfo = nfo_obj.info
nfo = sys.stderr.write


class MIRDatabase:
    """Machine Intelligence Resource Database"""

    database: Optional[dict[str, Any]]
    mir_file = JSONCache(MIR_PATH_NAMED)

    def __init__(self) -> None:
        self.read_from_disk()

    # @debug_monitor
    def add(self, resource: dict[str, Any]) -> None:
        """Merge pre-existing MIR entries, or add new ones
        :param element: _description_
        """
        parent_key = next(iter(resource))
        if self.database is not None:
            if self.database.get(parent_key, 0):
                self.database[parent_key] = {**self.database[parent_key], **resource[parent_key]}
            else:
                self.database[parent_key] = resource[parent_key]

    @mir_file.decorator
    def write_to_disk(self, data: Optional[dict] = None) -> None:  # pylint:disable=unused-argument
        """Save data to JSON file\n"""
        # from nnll.integrity import ensure_path
        try:
            os.remove(MIR_PATH_NAMED)
        except (FileNotFoundError, OSError) as error_log:
            nfo(f"MIR file not found before write, regenerating... {error_log}")
        self.mir_file.update_cache(self.database, replace=True)
        self.database = self.read_from_disk()
        nfo(f"Wrote {len(self.database)} lines to MIR database file.")

    @mir_file.decorator
    def read_from_disk(self, data: Optional[dict] = None) -> dict[str, Any]:
        """Populate mir database\n
        :param data: mir decorater auto-populated, defaults to None
        :return: dict of MIR data"""
        self.database = data
        return self.database

    # @debug_monitor
    def _stage_maybes(self, maybe_match: str, target: str, series: str, compatibility: str) -> List[str]:
        """Process a single value for matching against the target\n
        :param value: An unknown string value
        :param target: The search target
        :param series: MIR URI domain.arch.series identifier
        :param compatibility: MIR URI compatibility identifier\n
        (found value, path, sub-path,boolean for exact match)
        :return: _description_
        """
        results = []
        if isinstance(maybe_match, str):
            maybe_match = [maybe_match]
        for option in maybe_match:
            option_lower = option.lower()
            if option_lower == target:
                return [option, series, compatibility, True]
            elif target in option_lower:
                results.append([option, series, compatibility, False])
        return results

    @staticmethod
    def grade_maybes(matches: List[List[str]], target: str) -> list[str, str]:
        """Evaluate and select the best match from a list of potential matches\n
        :param matches: Possible matches to compare
        :param target: Desired entry to match
        :return: The closest matching dictionary elements
        """
        if not matches:
            return None
        min_gap = float("inf")
        best_match = None
        for match in matches:
            option, series, compatibility, _ = match
            if target in option or option in target:
                max_len = len(os.path.commonprefix([option, target]))
                gap = abs(len(option) - len(target)) + (len(option) - max_len)
                if gap < min_gap:
                    min_gap = gap
                    best_match = [series, compatibility]
        return best_match

    def ready_stage(self, maybe_match: str, target: str, series: str, compatibility: str) -> Optional[List[str]]:
        """Orchestrate match checking, return for exact matches, and create a queue of potential match
        :param maybe_match: The value of the requested search field
        :param target: The requested information
        :param series: Current MIR domain/arch/series tag
        :param compatibility: MIR compatibility tag
        :return: A list of exact matches or None
        """
        match_results = self._stage_maybes(maybe_match, target, series, compatibility)
        if next(iter(match_results), 0):
            if next(iter(match_results))[3]:
                return [series, compatibility]
            self.matches.extend(match_results)
        return None

    # @debug_monitor
    def find_path(self, field: str, target: str, sub_field: Optional[str] = None) -> list[str]:
        """Retrieve MIR path based on nested value search\n
        :param field: Known field to look within
        :param target: Search pattern for field
        :return: A list or string of the found tag
        :raises KeyError: Target string not found
        """
        target = target.lower()
        self.matches = []

        for series, comp in self.database.items():
            for compatibility, fields in comp.items():
                maybe_match = fields.get(field)
                if maybe_match is not None:
                    if isinstance(maybe_match, dict) and isinstance(next(iter(maybe_match.keys()), None), int):
                        for _, sub_field in maybe_match.items():
                            result = self.ready_stage(sub_field, target, series, compatibility)
                            if result:
                                return result
                    else:
                        result = self.ready_stage(maybe_match, target, series, compatibility)
                        if result:
                            return result

        best_match = self.grade_maybes(self.matches, target)
        if best_match:
            # dbug(best_match)
            return best_match
        else:
            nfo(f"Query '{target}' not found when searched {len(self.database)}'{field}' options")
            return None


def main(mir_db: Callable = MIRDatabase()) -> None:
    """Build the database"""
    from mir.automata import gen_diffusers, gen_torch_dtype, gen_schedulers, build_mir_lora, build_mir_custom, build_mir_additional

    gen_diffusers(mir_db)
    gen_torch_dtype(mir_db)
    gen_schedulers(mir_db)
    build_mir_lora(mir_db)
    build_mir_custom(mir_db)
    build_mir_additional(mir_db)
    mir_db.write_to_disk()


if __name__ == "__main__":
    import sys

    sys.path.append(os.getcwd())
    main()
