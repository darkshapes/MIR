# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


def main():
    import mir.maid
    from mir.maid import main as mir_main

    mir_main()
    from mir.inspect.tasks import main

    main()
    from mir.inspect.tasks import pipe

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
