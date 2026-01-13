# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import os

from mir.json_io import read_json_file
from logging import DEBUG, INFO, Logger

NFO = Logger(INFO).info
DBUQ = Logger(DEBUG).debug

ROOT_PATH = os.path.dirname(__file__)
MIR_PATH_NAMED = os.path.join(ROOT_PATH, "mir.json")
BREAKING = read_json_file(os.path.join(ROOT_PATH, "spec", "regex.json"))["breaking"]
SEARCH = read_json_file(os.path.join(ROOT_PATH, "spec", "regex.json"))["search"]
PARAMETERS = read_json_file(os.path.join(ROOT_PATH, "spec", "regex.json"))["parameters"]
SEMANTIC = read_json_file(os.path.join(ROOT_PATH, "spec", "regex.json"))["semantic"]
SUFFIX = read_json_file(os.path.join(ROOT_PATH, "spec", "regex.json"))["suffix"]
IGNORE = read_json_file(os.path.join(ROOT_PATH, "spec", "regex.json"))["ignore"]
