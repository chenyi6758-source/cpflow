"""CLI smoke tests."""

from cpflow.cli import main


def test_cli_case9(tmp_path):
    csv_path = tmp_path / "trace.csv"
    png_path = tmp_path / "pv.png"
    rc = main(["case9", "--csv", str(csv_path), "--plot", str(png_path)])
    assert rc == 0
    assert csv_path.exists()
    assert png_path.exists()
    assert png_path.stat().st_size > 1000
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) > 10  # header + P-V points
    assert lines[0].startswith("lambda,")
