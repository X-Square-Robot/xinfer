"""tools/docs_i18n.py: Chinese counterparts of the documentation pages, their links and their checks."""

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "docs_i18n.py"
_SPEC = importlib.util.spec_from_file_location("docs_i18n", _TOOL)
docs_i18n = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(docs_i18n)

# Code must match; comments and text blocks may be translated.
ENGLISH = {
    "a.md": "# A\n\n## Run it\n\nSee [setup](b.md#setup-steps) and [below](#run-it).\n\n"
            "```bash\nrun  # start\n```\n\n```text\nclient -> engine\n```\n",
    "b.md": "# B\n\n## Setup steps\n\n| x | y |\n|---|---|\n| 1 | 2 |\n",
}
CHINESE = {
    "a.zh-CN.md": "# 甲\n\n## 运行（示例）\n\n见[安装](b.md#setup-steps)和[下文](#run-it)。\n\n"
                  "```bash\nrun  # 运行\n```\n\n```text\n客户端 -> 引擎\n```\n",
    "b.zh-CN.md": "# 乙\n\n## 安装步骤\n\n| x | y |\n|---|---|\n| 1 | 2 |\n",
}


def _docs(root, pages):
    (root / "docs").mkdir(exist_ok=True)
    for name, text in pages.items():
        (root / "docs" / name).write_text(text, encoding="utf-8")


def test_links_adds_switches_and_maps_anchors_to_the_chinese_headings(tmp_path):
    _docs(tmp_path, {**ENGLISH, **CHINESE})
    assert docs_i18n.violations(tmp_path)  # Chinese pages still link to English pages, no switches yet
    docs_i18n.rewrite(tmp_path)
    a = (tmp_path / "docs" / "a.zh-CN.md").read_text(encoding="utf-8")
    assert a.splitlines()[2] == "[English](a.md) | 简体中文"
    assert "[安装](b.zh-CN.md#安装步骤)" in a and "[下文](#运行示例)" in a
    assert (tmp_path / "docs" / "a.md").read_text().splitlines()[2] == "English | [简体中文](a.zh-CN.md)"
    assert docs_i18n.violations(tmp_path) == []
    assert docs_i18n.rewrite(tmp_path) == []  # idempotent


def test_each_rule_reports_its_violation(tmp_path):
    _docs(tmp_path, {**ENGLISH, **CHINESE, "c.md": "# C\n"})
    docs_i18n.rewrite(tmp_path)
    docs = tmp_path / "docs"
    b = (docs / "b.zh-CN.md").read_text(encoding="utf-8")
    (docs / "b.zh-CN.md").write_text(b.replace("## 安装步骤", "### 安装步骤") + "| 3 | 4 |\n", encoding="utf-8")
    a = (docs / "a.zh-CN.md").read_text(encoding="utf-8")
    (docs / "a.zh-CN.md").write_text(a.replace("run  #", "go  #") + "[b](b.md) [x](x.md) [y](b.zh-CN.md#nope)\n",
                                     encoding="utf-8")
    found = docs_i18n.violations(tmp_path)
    assert sorted(entry.split(": ", 1)[1].split(": ")[-1] for entry in found) == sorted([
        "no c.zh-CN.md",
        "headings differ from b.zh-CN.md",
        "3 table rows, b.zh-CN.md 4",
        "code differs from a.zh-CN.md",
        "links to the other language",
        "no such file",
        "no such heading",
    ])


def test_a_page_without_its_language_switch_is_reported(tmp_path):
    _docs(tmp_path, {"b.md": ENGLISH["b.md"], "b.zh-CN.md": CHINESE["b.zh-CN.md"]})
    assert sorted(docs_i18n.violations(tmp_path)) == [
        "docs/b.md: no language switch to b.zh-CN.md in the first 12 lines",
        "docs/b.zh-CN.md: no language switch to b.md in the first 12 lines",
    ]
