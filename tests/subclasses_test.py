# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from mir.gatherers.transformers import GatherLoop
from mir.json_io import write_json_file

transformers_packages = GatherLoop()

from mir.gatherers.diffusers import GatherLoop

diffusers_packages = GatherLoop()

packages = {"transformers": transformers_packages.model_db, "diffusers": diffusers_packages.model_db}

write_json_file(folder_path_named="tests", file_name=".test.json", data=packages)


# def test_two():
#     from transformers import AltCLIPModel
#     from torch import nn
#     from mir.lookups import find_nn_modules

#     modules = find_nn_modules(AltCLIPModel)
#     for name, module in modules.items():
#         nn.ModuleList(module)


# if __name__ == "__main__":
#     test_two()
