# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from mir.spec import mir_entry
from mir import NFO
from mir.maid import MIRDatabase


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
