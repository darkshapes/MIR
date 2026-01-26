# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os

from mir import ROOT_PATH
from mir.json_io import read_json_file

MIR_PATH_NAMED = os.path.join(ROOT_PATH, "mir.json")


DIFFUSERS_ADDS = read_json_file(os.path.join(ROOT_PATH, "data", "diffusers_adds.json"))
EXCLUSIONS = read_json_file(os.path.join(ROOT_PATH, "data", "exclusions.json"))
MIGRATIONS = read_json_file(os.path.join(ROOT_PATH, "data", "migrations.json"))
NN_FILTER = read_json_file(os.path.join(ROOT_PATH, "data", "nn_filter.json"))
PARAMETERS = read_json_file(os.path.join(ROOT_PATH, "data", "parameters.json"))
PIPE_MARKERS = read_json_file(os.path.join(ROOT_PATH, "data", "pipe_markers.json"))
TAG_SCRAPE = read_json_file(os.path.join(ROOT_PATH, "data", "tag_scrape.json"))
TRANSFORMERS_ADDS = read_json_file(os.path.join(ROOT_PATH, "data", "transformers_adds.json"))
COMPONENT_NAMES = read_json_file(os.path.join(ROOT_PATH, "data", "component_names.json"))
