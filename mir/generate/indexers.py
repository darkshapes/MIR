# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""
# pylint:disable=no-name-in-module

from mir import NFO
from mir.data import MIGRATIONS
from mir.maid import MIRDatabase
from mir.spec import mir_entry


def write_to_mir(new_data: dict, mir_db: MIRDatabase) -> None:
    """Generate MIR HF Hub model database
    :param new_data: Data for the MIR database
    :param mir_database: MIRDatabase instance
    """
    for series, comp_name in new_data.items():
        id_segment = series.split(".")
        for compatibility in comp_name:
            # dbug(id_segment)
            try:
                mir_db.add(
                    mir_entry(
                        domain=id_segment[0],
                        arch=id_segment[1],
                        series=id_segment[2],
                        comp=compatibility,
                        **new_data[series][compatibility],
                    ),
                )
            except IndexError:  # as error_log:
                NFO(f"Failed to create series: {series}  compatibility: {comp_name}  ")
                # dbug(error_log)


def migrations(repo_path: str):
    """Replaces old organization names in repository paths with new ones.\n
    :param repo_path: Original repository path containing old organization names
    :return: Updated repository path with new organization names"""

    repo_migrations = MIGRATIONS
    for old_name, new_name in repo_migrations.items():
        if old_name in repo_path:
            repo_path = repo_path.replace(old_name, new_name)
    return repo_path
