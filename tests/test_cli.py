import pytest

from hn_mri_prep.cli import main


def test_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "preprocess" in capsys.readouterr().out
