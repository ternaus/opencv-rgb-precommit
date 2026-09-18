from pathlib import Path

from opencv_rgb_policy.cli import main


def test_cli_reports_policy_violation(tmp_path: Path, capsys) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "import cv2\nimage = cv2.imread(path, cv2.IMREAD_COLOR)\nimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
        encoding="utf-8",
    )

    assert main([str(source)]) == 1
    output = capsys.readouterr().err
    assert "OPCV001" in output


def test_cli_accepts_clean_file(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("import cv2\nimage = cv2.imread(path, cv2.IMREAD_COLOR_RGB)\n", encoding="utf-8")

    assert main([str(source)]) == 0
