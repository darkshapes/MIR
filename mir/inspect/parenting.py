# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import List, Optional


def class_parent(code_name: str, pkg_name: str) -> Optional[List[str]]:
    """Retrieve the folder path within a class. Only returns if it is a valid path in the system\n
    ### NOTE: in most cases `__module__` makes this redundant
    :param code_name: The internal name for the model in the third-party API.
    :param pkg_name: The API Package
    :return: A list corresponding to the path of the model, or None if not found
    :raises KeyError: for invalid pkg_name
    """
    import os
    from importlib import import_module

    pkg_paths = {
        "diffusers": "pipelines",
        "transformers": "models",
    }
    folder_name = code_name.replace("-", "_")
    pkg_name = pkg_name.lower()
    folder_path = pkg_paths[pkg_name]
    package_obj = import_module(pkg_name)
    folder_path_named = [folder_path, folder_name]
    pkg_folder = os.path.dirname(getattr(package_obj, "__file__"))
    # dbuq(os.path.exists(os.path.join(pkg_folder, *folder_path_named)))
    if os.path.exists(os.path.join(pkg_folder, *folder_path_named)) is True:
        import_path = [pkg_name]
        import_path.extend(folder_path_named)
        return import_path
