"""tools/check_repo.py reports each kind of violation."""

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_repo.py"
_SPEC = importlib.util.spec_from_file_location("check_repo", _TOOL)
check_repo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_repo)


def test_each_rule_reports_its_violation(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "model_history.md").write_text("results of each revision\n")
    (tmp_path / "notes.md").write_text("data in /m" + "nt/runs/1\n")  # split so this file does not match
    (tmp_path / "big.bin").write_bytes(b"\0" * (check_repo.MAX_FILE_BYTES + 1))
    (tmp_path / "module.py").write_text("x = 1  # fixed in group/project" + "#3\n")
    (tmp_path / "clean.py").write_text("x = 1  # the cache is per lane\n")
    found = check_repo.violations(tmp_path)
    assert {entry.split(":")[0] for entry in found} == {"docs/model_history.md", "notes.md", "big.bin", "module.py"}
