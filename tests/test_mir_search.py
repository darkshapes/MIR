import pytest


def test_mir_maid():
    from mir.mir_maid import MIRDatabase, main
    from mir.json_cache import MIR_PATH_NAMED
    import json

    expected = {"empty": "101010101010101010"}
    mir_db = MIRDatabase()
    mir_db.database = expected
    mir_db.write_to_disk()
    with open(MIR_PATH_NAMED, "r", encoding="UTF-8") as f:
        result = json.load(f)

    assert result == expected


def test_restore_mir():
    from mir.mir_maid import MIRDatabase, main
    from mir.json_cache import MIR_PATH_NAMED
    import json

    mir_db = MIRDatabase()
    mir_db.database.pop("empty")
    main(mir_db)
    expected = mir_db.database
    with open(MIR_PATH_NAMED, "r", encoding="UTF-8") as f:
        result = json.load(f)
    for tag, compatibility in result.items():
        for comp, field in compatibility.items():
            for header, definition in field.items():
                if isinstance(definition, dict):
                    for key in definition:
                        if len(key) > 1:
                            assert field[header][key] == expected[tag][comp][header][key]
                        else:
                            assert field[header][key] == expected[tag][comp][header][int(key)]
                else:
                    assert field[header] == expected[tag][comp][header]

    print(mir_db.database)


@pytest.fixture
def mock_test_database():
    from mir.mir_maid import MIRDatabase

    mir_db = MIRDatabase()

    return mir_db


def test_grade_char_match(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="table-cascade")
    assert result == ["info.unet.stable-cascade", "base"]


def test_grade_similar_match(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="able-cascade-")
    assert result == ["info.unet.stable-cascade", "prior"]


def test_grade_field_change(mock_test_database):
    result = mock_test_database.find_path(field="pkg", target="parler_tts")
    assert result == ["info.art.parler-tts", "tiny-v1"]


def test_grade_letter_case_change(mock_test_database):
    result = mock_test_database.find_path(field="pkg", sub_field=0, target="AuDiOCrAfT.MoDeLs")
    assert result == ["info.art.audiogen", "medium-1.5b"]


def test_grade_cannot_find(mock_test_database):
    test = "asdjfd"
    field = ("pkg",)
    with pytest.raises(KeyError) as excinfo:
        result = mock_test_database.find_path(field=field, target=test)
    assert str(excinfo.value) == f"\"Query '{test}' not found when searched {len(mock_test_database.database)}'{field}' options\""
