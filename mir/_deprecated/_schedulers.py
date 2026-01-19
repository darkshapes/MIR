# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import re
from importlib import import_module

from mir.generate.diffusers import IMPORT_STRUCTURE
from mir.maid import MIRDatabase
from mir.spec import mir_entry


def tag_scheduler(series_name: str) -> tuple[str, str]:
    """Create a mir label from a scheduler operation\n
    :param class_name: Known period-separated prefix and model type
    :return: The assembled mir tag with compatibility pre-separated"""

    comp_name = None
    patterns = [r"Schedulers", r"Multistep", r"Solver", r"Discrete", r"Scheduler"]
    for scheduler in patterns:
        compiled = re.compile(scheduler)
        match = re.search(compiled, series_name)
        if match:
            comp_name = match.group()
            comp_name = comp_name.lower()
            break
    for pattern in patterns:
        series_name = re.sub(pattern, "", series_name)
    series_name.lower()
    assert series_name is not None, "Expected series tag but got None"
    assert comp_name is not None, "Expected compatibility tag but got None"
    return series_name, comp_name


def add_schedulers(mir_db: MIRDatabase):
    """Create mir info database"""

    for class_name in IMPORT_STRUCTURE["schedulers"]:
        if class_name != "SchedulerMixin":
            series_name, comp_name = tag_scheduler(class_name)
            class_obj = import_module("diffusers.schedulers")
            class_path = getattr(class_obj, class_name).__module__
            mir_db.add(
                mir_entry(
                    domain="ops",
                    arch="scheduler",
                    series=series_name,
                    comp=comp_name.lower(),
                    pkg={
                        0: {
                            "diffusers": class_name,
                            "module_path": class_path,
                        },
                    },
                )
            )

    class_name = "KarrasDiffusionSchedulers"
    series_name, comp_name = tag_scheduler(class_name)
    class_obj = import_module("diffusers.schedulers.scheduling_utils")
    class_path = getattr(class_obj, class_name).__module__
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="scheduler",
            series=series_name,
            comp=comp_name,
            pkg={
                0: {
                    "diffusers": class_name,
                    "module_path": class_path,
                },
            },
        ),
    )
