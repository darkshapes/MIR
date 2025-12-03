# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from mir.__main__ import main
from mir.maid import MIRDatabase


def test_task_and_pipe():
    mir_db = MIRDatabase()
    assert main(mir_db) is not None
