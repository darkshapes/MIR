#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0 */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import pytest
from mir.inspect.classes import extract_init_params


def test_root_class_with_builtin_types():
    class DummyInitModule:
        def __init__(self, flag: bool, count: int):
            pass

    expected_output = {}

    result = extract_init_params(DummyInitModule)
    assert result == expected_output


if __name__ == "__main__":
    import pytest

    pytest.main(["-vv", __file__])
