# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""神经网络的数据注册"""

# pylint: disable=possibly-used-before-assignment, line-too-long
import os
from typing import Any, List, Optional

from mir import MIR_PATH_NAMED
from mir.json_io import read_json_file, write_json_file


class MIRDatabase:
    """Machine Intelligence Resource Database Object
    Database search and read/write operations"""

    def __init__(self, database: dict | None = None) -> None:
        from json.decoder import JSONDecodeError
        from mir import DBUQ

        if not database:
            self.database = {"expected": "data"}
            try:
                self.read_from_disk()
            except JSONDecodeError as error_log:
                DBUQ(error_log)
                self.database = {}

    def add(self, resource: dict[str, Any]) -> None:
        """Merge pre-existing MIR entries, or add new ones
        :param resource: Entry to apply
        """
        parent_key = next(iter(resource))
        if self.database is not None:
            if self.database.get(parent_key, 0):
                self.database[parent_key] = self.database[parent_key] | resource[parent_key]
            else:
                self.database[parent_key] = resource[parent_key]

    def write_to_disk(self, data: Optional[dict] = None) -> None:  # pylint:disable=unused-argument
        """Save data to JSON file\n"""

        from mir import NFO

        if not os.path.exists(MIR_PATH_NAMED):
            mode = "x"
        else:
            mode = "w"
        # except (FileNotFoundError, OSError) as error_log:
        #     nfo(f"MIR file not found before write, regenerating... {error_log}")

        write_json_file(os.path.dirname(MIR_PATH_NAMED), file_name="mir.json", data=self.database, mode=mode)
        written_data = self.read_from_disk()
        NFO(f"Wrote {len(written_data)} lines to MIR database file.")
        self.database = written_data

    def read_from_disk(self, data: Optional[dict] = None) -> dict[str, Any]:
        """Populate mir database\n
        :param data: mir decorator auto-populated, defaults to None
        :return: dict of MIR data"""
        if not os.path.exists(MIR_PATH_NAMED):
            self.write_to_disk({})
            return self.database
        else:
            self.database = read_json_file(MIR_PATH_NAMED)
            return self.database

    def _stage_maybes(self, maybe_match: str, target: str, series: str, compatibility: str) -> list[str | bool]:
        """Process a single value for matching against the target\n
        :param value: An unknown string value
        :param target: The search target
        :param series: MIR URI domain.arch.series identifier
        :param compatibility: MIR URI compatibility identifier\n
        (found value, path, sub-path,boolean for exact match)
        :return: A list of likely options and their MIR paths"""
        import re

        from mir import SEARCH

        results = []
        if isinstance(maybe_match, str):
            maybe_match: list[str] = [maybe_match]
        elif isinstance(maybe_match, dict):
            if isinstance(next(iter(maybe_match)), int):
                maybe_match = list(maybe_match.values())
            else:
                maybe_match = list(maybe_match.keys())
        for option in maybe_match:
            option_lower = re.sub(SEARCH, "", option.lower())
            target = re.sub(SEARCH, "", target.lower())
            if option_lower:
                if option_lower:
                    if option_lower in target:
                        return [option, series, compatibility, True]
                    elif target in option_lower:
                        results.append([option, series, compatibility, False])
        return results

    @staticmethod
    def grade_maybes(matches: List[List[str]], target: str) -> list[str] | None:
        """Evaluate and select the best match from a list of potential matches\n
        :param matches: Possible matches to compare
        :param target: Desired entry to match
        :return: The closest matching dictionary elements
        """
        from decimal import Decimal
        from math import isclose

        if not matches:
            return None
        min_gap = float("inf")
        best_match = None
        for match in matches:
            option, series, compatibility, _ = match
            option = option.replace("_", "").replace("-", "").replace(".", "").lower()
            if target in option or option in target:
                max_len = len(os.path.commonprefix([option, target]))
                gap = Decimal(str(abs(len(option) - len(target)) + (len(option) - max_len))) * Decimal("0.1")
                if gap < min_gap and isclose(gap, 0.9, rel_tol=15e-2):  # 15% variation, 5% error margin, 45% buffer below fail
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
        if maybe_match:
            match_results = self._stage_maybes(maybe_match, target, series, compatibility)
            if next(iter(match_results), 0):
                if next(iter(match_results))[3]:
                    return [series, compatibility]
                self.matches.extend(match_results)
        return None

    def find_tag(self, field: str, target: str, sub_field: Optional[str] = None, domain: None | str = None) -> list[str]:
        """Retrieve MIR path based on nested value search\n
        :param field: Known field to look within
        :param target: Search pattern for field
        :param sub_field: A Second field level to investigate into (ex, field pkg, sub_field diffusers)
        :return: A list or string of the found tag
        :raises KeyError: Target string not found
        """
        import re
        from mir import NFO

        parameters = r"-gguf|-exl2|-exl3|-onnx|-awq|-mlx|-ov"  #
        target = target.lower().strip("-")
        target = re.sub(parameters, "", target)
        self.matches = []

        for series, comp in self.database.items():
            if (not domain) or series.startswith(domain):
                for compatibility, fields in comp.items():
                    if maybe_match := fields.get(field):
                        if isinstance(maybe_match, dict) and str(next(iter(maybe_match.keys()), None)).isnumeric():  #  is a dictionary with a number
                            for _, sub_maybe in maybe_match.items():
                                if result := self.ready_stage(sub_maybe.get(sub_field, list(sub_maybe)), target, series, compatibility):
                                    return result
                        else:
                            if result := self.ready_stage(maybe_match, target, series, compatibility):
                                return result

        if best_match := self.grade_maybes(self.matches, target):
            return best_match
        else:
            NFO(f"Query '{target}' not found when {len(self.database)}'{field}' options searched\n")
            return None
