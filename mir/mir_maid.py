### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""神经网络的数据注册"""

# pylint: disable=possibly-used-before-assignment, line-too-long
from typing import Any, Callable, Iterable, List, Optional
import os

# from mir.constants import CueType
from nnll.monitor.file import debug_monitor, dbug, nfo
from mir.json_cache import JSONCache, MIR_PATH_NAMED  # pylint:disable=no-name-in-module


class MIRDatabase:
    """Machine Intelligence Resource Database"""

    database: Optional[dict[str, Any]]
    mir_file = JSONCache(MIR_PATH_NAMED)

    def __init__(self) -> None:
        self.read_from_disk()

    @debug_monitor
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
            dbug(f"MIR file not found before write, regenerating... {error_log}")
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

    @debug_monitor
    def _ready_value(self, value: str, target: str, series: str, compatibility: str) -> List[str]:
        """Process a single value for matching against the target\n
        :param value: An unknown string value
        :param target: The search target
        :param series: MIR URI domain.arch.series identifier
        :param compatibility: MIR URI compatibility identifier\n
        (found value, path, sub-path,boolean for exact match)
        :return: _description_
        """
        results = []
        if isinstance(value, str):
            value = [value]
        for option in value:
            option_lower = option.lower()
            if option_lower == target:
                return [option, series, compatibility, True]
            elif target in option_lower:
                results.append([option, series, compatibility, False])
        return results

    @staticmethod
    def grade_char_match(matches: List[List[str]], target: str) -> list[str, str]:
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
            print(_)
            if target in option or option in target:
                max_len = len(os.path.commonprefix([option, target]))
                gap = abs(len(option) - len(target)) + (len(option) - max_len)
                if gap < min_gap:
                    min_gap = gap
                    best_match = [series, compatibility]
        return best_match

    @debug_monitor
    def find_path(self, field: str, target: str, sub_field: Optional[str] = None) -> list[str]:
        """Retrieve MIR path based on nested value search\n
        :param field: Known field to look within
        :param target: Search pattern for field
        :return: A list or string of the found tag
        :raises KeyError: Target string not found
        """
        target = target.lower()
        matches = []

        def process_value(value, series, compatibility):
            match_results = self._ready_value(value, target, series, compatibility)
            if next(iter(match_results), 0):
                if next(iter(match_results))[3]:
                    return [series, compatibility]
                matches.extend(match_results)
            return None

        for series, comp in self.database.items():
            for compatibility, fields in comp.items():
                value = fields.get(field)
                if value is not None:
                    if isinstance(value, dict) and isinstance(next(iter(value.keys()), None), int):
                        for _, sub_field in value.items():
                            result = process_value(sub_field, series, compatibility)
                            if result:
                                return result
                    else:
                        result = process_value(value, series, compatibility)
                        if result:
                            return result

        best_match = self.grade_char_match(matches, target)
        if best_match:
            dbug(best_match)
            return best_match
        else:
            dbug(f"Query '{target}' not found when searched {len(self.database)}'{field}' options")
            return None


def main(mir_db: Callable = MIRDatabase()) -> None:
    """Build the database"""
    from mir.automata import mir_diffusion, mir_dtype, mir_scheduler, build_mir_lora, build_mir_custom, build_mir_additional

    mir_diffusion(mir_db)
    mir_dtype(mir_db)
    mir_scheduler(mir_db)
    build_mir_lora(mir_db)
    build_mir_custom(mir_db)
    build_mir_additional(mir_db)
    mir_db.write_to_disk()


if __name__ == "__main__":
    import sys

    sys.path.append(os.getcwd())
    main()
