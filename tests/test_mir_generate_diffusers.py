def test_info_key_exists_and_library_is_not_nested():
    from mir.generate.diffusers.harvest import HarvestLoop

    Mir = HarvestLoop().db.db

    # print(Mir)
