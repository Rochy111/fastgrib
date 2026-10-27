"""fastgrib.warm() and `fastgrib -warm`: the sample shipped in the package
(a field of every packing) decodes on every path."""
import pytest

import fastgrib
from fastgrib.__main__ import main


def test_warm():
    assert fastgrib.warm() >= 0


def test_warm_on_the_command_line(capsys):
    assert main(['-warm']) == 0
    assert 'decoders compiled' in capsys.readouterr().out


def test_command_line_needs_a_file():
    with pytest.raises(SystemExit):
        main([])
