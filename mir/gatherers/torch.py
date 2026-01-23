# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import re

import torch

from mir import DBUQ
from mir.maid import MIRDatabase
from mir.spec import mir_entry


def slice_number(text: str) -> int | float | str:
    """Separate a numeral value appended to a string\n
    :return: Converted value as int or float, or unmodified string
    """
    for index, char in enumerate(text):  # Traverse forwards
        if char.isdigit():
            numbers = text[index:]
            if "." in numbers:
                return float(numbers)
            try:
                return int(numbers)
            except ValueError:
                return numbers
    return text


def add_mir_dtype(mir_db: MIRDatabase):
    """Create mir info database"""

    available_dtypes: list[torch.dtype] = [dtype for dtype in torch.__dict__.values() if isinstance(dtype, torch.dtype)]
    series_name = "_"
    for precision in available_dtypes:
        dep_name, class_name = str(precision).split(".")
        if "_" in class_name:
            comp_name = class_name[0].upper() + "8_" + class_name.split("_")[1].upper()
            if comp_name.endswith("FN"):
                comp_name = comp_name[:-2]
        else:
            comp_name = class_name[0].upper() + str(slice_number(class_name))
        variant_name = class_name.replace("bfloat", "bf").replace("float", "fp")
        DBUQ(variant_name)
        patterns = [r"complex", r"bits", r"quint", r"uint", r"int", r"bfloat", r"float", r"bool"]
        for precision_name in patterns:
            compiled = re.compile(precision_name)
            dtype = re.search(compiled, class_name)
            if dtype:
                series_name = dtype.group()
                break

        mir_db.add(
            mir_entry(
                domain="ops",
                arch="precision",
                series=series_name,
                comp=comp_name,
                pkg={0: {dep_name.lower(): {class_name.lower(): {"variant": variant_name}}}},
            )
        )
