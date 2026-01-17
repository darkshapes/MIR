# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
from mir.tag import tag_model_from_repo


# def test_param_no_delimiter_version():BAH
#     result = make_mir_tag("xyz1b")
#     assert result == ("xyz", "*")
#     print(result)


def test_split_hyphenated():
    result = tag_model_from_repo("xyz-15b")
    assert result == ("xyz", "*")
    print(result)


# def test_split_dot(): BAH
#     result = make_mir_tag("xyz.15b")
#     assert result == ("xyz", "*")


def test_split_dot_version():
    assert tag_model_from_repo("xyz1.0") == ("xyz1", "*")


def test_split_hyphen_version():
    assert tag_model_from_repo("xyz1-0") == ("xyz1-0", "*")


def test_split_hyphen_v_version():
    assert tag_model_from_repo("xyzv1-0") == ("xyzv1-0", "*")


def test_no_split():
    assert tag_model_from_repo("flux.1-dev") == ("flux1-dev", "*")


def test_no_split_again():
    assert tag_model_from_repo("blipdiffusion") == ("blipdiffusion", "*")


def test_no_version_dot_numeric_and_diffusers():
    assert tag_model_from_repo("EasyAnimateV5.1-7b-zh-diffusers") == ("easyanimatev5-zh", "diffusers")
