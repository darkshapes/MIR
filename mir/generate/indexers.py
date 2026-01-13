# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""
# pylint:disable=no-name-in-module

from mir.generate import MIGRATIONS


def migrations(repo_path: str):
    """Replaces old organization names in repository paths with new ones.\n
    :param repo_path: Original repository path containing old organization names
    :return: Updated repository path with new organization names"""

    repo_migrations = MIGRATIONS
    for old_name, new_name in repo_migrations.items():
        if old_name in repo_path:
            repo_path = repo_path.replace(old_name, new_name)
    return repo_path
