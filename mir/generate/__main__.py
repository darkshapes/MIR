# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from mir.maid import MIRDatabase
from mir.generate.tasks import TaskAnalyzer
from typing import Callable


def run_task() -> None:
    main()


def pipe(mir_db: MIRDatabase) -> MIRDatabase:
    import argparse
    import asyncio
    from sys import modules as sys_modules

    if "pytest" not in sys_modules:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Infer pipe components from Diffusers library and attach them to an existing MIR database.\nOffline function.",
            usage="mir-pipe",
            epilog="Can be run automatically with `python -m nnll.mir.maid` Should only be used after `mir-maid`.\n\nOutput:\n    INFO     ('Wrote #### lines to MIR database file.',)",
        )
        parser.parse_args()

    from mir.generate.automata import assimilate

    if not mir_db:
        mir_db = MIRDatabase()

    tasker = TaskAnalyzer()
    pipe_tuple = asyncio.run(tasker.detect_pipes(mir_db))
    assimilate(mir_db, [pipe for pipe in pipe_tuple])
    mir_db.write_to_disk()
    return mir_db


# if __name__ == "__main__":
#     pipe()


def main():
    # import ordered to prevent file lock
    import mir.maid
    from mir.maid import main as mir_main

    mir_main()
    from mir.generate.tasks import main

    main()
    from mir.generate.tasks import pipe

    pipe()

    import os
    import shutil

    try:
        os.remove("mir.json")
    except FileNotFoundError:
        pass
    shutil.copy2(os.path.join(os.path.dirname(mir.maid.__file__), "mir.json"), os.path.join(os.getcwd(), "mir.json"))


if __name__ == "__main__":
    main()


def main(mir_db: MIRDatabase | None = None) -> MIRDatabase:
    """Parse arguments to feed to dict header reader"""
    import argparse
    import asyncio
    from mir.generate.automata import assimilate
    from sys import modules as sys_modules

    if "pytest" not in sys_modules:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Scrape the task classes from currently installed libraries and attach them to an existing MIR database.\nOffline function.",
            usage="mir-tasks",
            epilog="Can be run automatically with `python -m nnll.mir.maid` Should only be used after `mir-maid`.\n\nOutput:\n    INFO     ('Wrote #### lines to MIR database file.',)",
        )
        parser.parse_args()

    if not mir_db:
        mir_db = MIRDatabase()

    tasker = TaskAnalyzer()
    task_tuple = asyncio.run(tasker.detect_tasks(mir_db))

    assimilate(mir_db, [task for task in task_tuple])

    mir_db.write_to_disk()
    return mir_db


def main(mir_db: Callable | None = None, remake: bool = True) -> None:
    """Build the database"""
    from sys import modules as sys_modules

    if __name__ != "__main__" and "pytest" not in sys_modules:  #
        import argparse

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Build a custom MIR model database from the currently installed system environment.\nOffline function.",
            usage="mir-maid",
            epilog="""Does NOT include results of `mir-task` and `mir-pipe`. These commands should be run separately. Output:
            2025-08-03 14:22:47 INFO     ('Wrote 0 lines to MIR database file.',)
            2025-08-03 14:22:47 INFO     ('Wrote #### lines to MIR database file.',)""",
        )
        parser.add_argument(
            "-r",
            "--remake_off",
            action="store_true",
            default=False,
            help="Prevent erasing and remaking the MIR database file (default: False, always start from a completely empty MIR file)",
        )

        args = parser.parse_args()
        remake = not args.remake_off

    from mir.generate.automata import (
        add_mir_audio,
        add_mir_diffusion,
        add_mir_dtype,
        add_mir_llm,
        add_mir_lora,
        add_mir_schedulers,
        add_mir_vae,
        hf_pkg_to_mir,
        mir_update,
    )
    from mir.json_io import write_json_file

    if remake:
        os.remove(MIR_PATH_NAMED)
        folder_path_named = os.path.dirname(MIR_PATH_NAMED)
        mode = "x"
    else:
        mode = "w"
    write_json_file(folder_path_named, file_name="mir.json", data={"expected": "data"}, mode=mode)
    mir_db = MIRDatabase()
    mir_db.database.pop("expected", {})
    hf_pkg_to_mir(mir_db)
    add_mir_dtype(mir_db)
    add_mir_schedulers(mir_db)
    add_mir_lora(mir_db)
    add_mir_audio(mir_db)
    add_mir_diffusion(mir_db)
    add_mir_llm(mir_db)
    add_mir_vae(mir_db)
    mir_db.write_to_disk()
    mir_db = MIRDatabase()
    mir_db = MIRDatabase()
    mir_update(mir_db)
    mir_db.write_to_disk()


if __name__ == "__main__":
    remake: bool = True
    tasks = True
    pipes = True

    from sys import modules as sys_modules

    if "pytest" not in sys_modules:  #
        import argparse

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Build a custom MIR model database from the currently installed system environment.\nOffline function.",
            usage="python -m nnll.mir.maid",
            epilog="""Includes `mir-task` and `mir-pipe` by default. Output:
            2025-08-15 19:41:18 INFO     ('Wrote 0 lines to MIR database file.',)
            2025-08-15 19:38:48 INFO     ('Wrote ### lines to MIR database file.',)
                                INFO     ('Wrote ### lines to MIR database file.',)
                                INFO     ('Wrote ### lines to MIR database file.',)""",
        )
        parser.add_argument(
            "-r",
            "--remake_off",
            action="store_true",
            default=False,
            help="Don't erase and remake the MIR database (default: False)",
        )
        parser.add_argument(
            "-t",
            "--tasks_off",
            action="store_true",
            default=False,
            help="Don't append task information to the MIR database (default: False)",
        )
        parser.add_argument(
            "-p",
            "--pipes_off",
            action="store_true",
            default=False,
            help="Don't append pipeline information to the MIR database (default: False)",
        )

        args = parser.parse_args()
        remake = not args.remake_off
        tasks = not args.tasks_off
        pipes = not args.pipes_off

    main(remake=remake)

    from mir.generate.tasks import pipe, run_task

    mir_db = run_task()
    pipe(mir_db)


def main(mir_db: MIRDatabase = None):
    """Parse arguments to feed to dict header reader"""
    import argparse
    import asyncio
    from mir.automata import assimilate
    from sys import modules as sys_modules

    if "pytest" not in sys_modules:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Scrape the task classes from currently installed libraries and attach them to an existing MIR database.\nOffline function.",
            usage="mir-tasks",
            epilog="Can be run automatically with `python -m nnll.mir.maid` Should only be used after `mir-maid`.\n\nOutput:\n    INFO     ('Wrote #### lines to MIR database file.',)",
        )
        parser.parse_args()

    if not mir_db:
        mir_db = MIRDatabase()

    auto_pkg = TaskAnalyzer()
    task_tuple = asyncio.run(auto_pkg.detect_tasks(mir_db))

    assimilate(mir_db, [task for task in task_tuple])

    mir_db.write_to_disk()
    return mir_db


def run_task():
    main()


def pipe(mir_db: MIRDatabase = None):
    import argparse
    import asyncio
    from sys import modules as sys_modules

    if "pytest" not in sys_modules:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Infer pipe components from Diffusers library and attach them to an existing MIR database.\nOffline function.",
            usage="mir-pipe",
            epilog="Can be run automatically with `python -m nnll.mir.maid` Should only be used after `mir-maid`.\n\nOutput:\n    INFO     ('Wrote #### lines to MIR database file.',)",
        )
        parser.parse_args()

    from mir.automata import assimilate

    if not mir_db:
        mir_db = MIRDatabase()

    auto_pkg = TaskAnalyzer()
    pipe_tuple = asyncio.run(auto_pkg.detect_pipes(mir_db))
    assimilate(mir_db, [pipe for pipe in pipe_tuple])
    mir_db.write_to_disk()
    return mir_db


if __name__ == "__main__":
    pipe()
