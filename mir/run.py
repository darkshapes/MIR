#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0*/ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->


from mir.maid import MIRDatabase
from mir.inspect.tasks import TaskAnalyzer


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

    tasker = TaskAnalyzer()
    task_tuple = asyncio.run(tasker.detect_tasks(mir_db))

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

    tasker = TaskAnalyzer()
    pipe_tuple = asyncio.run(tasker.detect_pipes(mir_db))
    assimilate(mir_db, [pipe for pipe in pipe_tuple])
    mir_db.write_to_disk()
    return mir_db


# if __name__ == "__main__":
#     pipe()
