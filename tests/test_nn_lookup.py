# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable
from mir.lookups import get_source_of, nn_source_tree, get_import_chain
from mir.gatherers.transformers import AUTO_MAP
import torch
from torch import nn
from transformers import Aimv2TextConfig
from mir.json_io import write_json_file


@torch.no_grad
def test_lookups():
    lookups = []
    for config, model in AUTO_MAP.items():
        if isinstance(model, tuple):
            model: Callable = model[0]  # type: ignore
        try:
            model_source = get_source_of(model)
        except AttributeError as _:
            print(model.__name__)
            continue
        try:
            if call_data := nn_source_tree(model_source):
                print(call_data)
                model_path = model.__module__
                try:
                    module_obj: Callable = get_import_chain(f"{model_path}.{call_data['class_name']}")
                except AttributeError as _:
                    print(model.__name__)
                    continue
                try:
                    config_obj = config()
                except (TypeError, ImportError) as _:
                    print(model.__name__)
                    continue
                if hasattr(config_obj, call_data["config_attribute"]):
                    config_attribute = getattr(config_obj, call_data["config_attribute"])
                elif call_data["class_name"] == "Aimv2EncoderLayer":
                    config_obj = Aimv2TextConfig()
                    config_attribute = getattr(config_obj, call_data["config_attribute"])
                try:
                    lookups.append(nn.ModuleList(module_obj(config_obj) for _ in range(config_attribute)))
                except TypeError as _:
                    print(f"error with {call_data['class_name']}")
                except AttributeError as _:
                    print(f"no attribute for with {call_data['class_name']} config.{config_attribute}")
                except KeyError as _:
                    print(f"no attribute for with {call_data['class_name']} config.{config_attribute}")
            print(model.__name__)
        except IndexError as _:
            print(model.__name__)
    with open("somesuch.txt", mode="w", encoding="utf-8") as i:
        i.write(str(lookups))


test_lookups()
