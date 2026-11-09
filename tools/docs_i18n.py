"""Bilingual documentation: every page exists in English and Simplified Chinese, kept in step.

Pages are ``README.md`` and ``docs/**/*.md``; the Chinese page of ``X.md`` is ``X.zh-CN.md`` next to it. A merge
request that changes a page changes its counterpart too (CONTRIBUTING.md, "Repository contents").

``check`` (run by CI) reports, and exits with status 1 on:

- a page without its counterpart;
- a page whose first lines do not link to its counterpart (the language switch);
- a pair whose headings (levels, in order), table rows or code differ: fenced blocks pair up in order with the same
  language, and their code is equal once comments are removed (``text`` and unlabeled blocks, such as diagrams and
  output, may be translated);
- a relative link to a missing file, or to an anchor that is not a heading of the linked page;
- a link to a page of the other language, other than the language switch.

``links`` adds missing language switches and rewrites each Chinese page's links to English pages into links to their
Chinese counterparts, mapping every anchor to the counterpart's heading at the same position. Translate with the
English link targets in place, then run it.

Anchors follow GitLab (and GitHub): lowercase, characters other than letters, digits, ``_``, ``-`` and spaces removed,
each space to a hyphen, ``-1``, ``-2``, ... appended to repeats.

usage: python tools/docs_i18n.py {check,links} [root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ZH = ".zh-CN.md"
SWITCH_LINES = 12  # the language switch is within a page's first lines
FENCE = re.compile(r"\s*(```|~~~)")
HEADING = re.compile(r"(#{1,6})\s+(.*?)\s*$")
LINK = re.compile(r"\]\(([^)\s]+)\)|\b(?:href|src)=\"([^\"]+)\"")
INLINE_CODE = re.compile(r"`[^`]*`")


def counterpart(page: Path) -> Path:
    """The other language's page of ``page``."""
    if page.name.endswith(ZH):
        return page.with_name(page.name[: -len(ZH)] + ".md")
    return page.with_name(page.stem + ZH)


def pages(root: Path) -> list[Path]:
    """Every documentation page, English and Chinese."""
    found = [root / "README.md", root / "README.zh-CN.md", *(root / "docs").rglob("*.md")]
    return sorted({page for page in found if page.is_file()})


def parse(text: str):
    """``(headings [(level, text)], table rows, code blocks [(language, lines)], prose [(line number, line)])``;
    code blocks excluded from the rest."""
    headings, rows, blocks, prose = [], 0, [], []
    fence, language, block = None, "", []
    for number, line in enumerate(text.splitlines(), 1):
        if fence:
            if line.strip().startswith(fence):
                blocks.append((language, block))
                fence = None
            else:
                block.append(line)
            continue
        if match := FENCE.match(line):
            fence, language, block = match.group(1), line.strip()[3:].strip(), []
            continue
        prose.append((number, line))
        if match := HEADING.match(line):
            headings.append((len(match.group(1)), match.group(2)))
        elif line.lstrip().startswith("|"):
            rows += 1
    return headings, rows, blocks, prose


def code(block) -> tuple[str, str | None]:
    """What a pair's code block must share: its language and, unless it is ``text`` or unlabeled, its code without
    ``#`` comments."""
    language, lines = block
    if language in ("", "text"):
        return language, None
    kept = [re.sub(r"\s+#\s.*$", "", line).rstrip() for line in lines if not line.lstrip().startswith("#")]
    return language, "\n".join(kept)


def anchors(text: str) -> list[str]:
    """The heading anchors of a page, in order."""
    seen, out = {}, []
    for _, heading in parse(text)[0]:
        heading = re.sub(r"<[^>]*>", "", re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)).lower()
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        out.append(slug if count == 0 else f"{slug}-{count}")
    return out


def links(text: str):
    """``(line number, start, end, target)`` of every relative link outside code; ``start:end`` spans the target
    within its line."""
    for number, line in parse(text)[3]:
        code = [match.span() for match in INLINE_CODE.finditer(line)]
        for match in LINK.finditer(line):
            group = 1 if match.group(1) is not None else 2
            target = match.group(group)
            if re.match(r"[a-z][\w+.-]*:", target) or any(a <= match.start() < b for a, b in code):
                continue
            yield number, match.start(group), match.end(group), target


def _resolve(page: Path, target: str) -> tuple[Path, str]:
    path, _, anchor = target.partition("#")
    return ((page.parent / path).resolve() if path else page.resolve()), anchor


def _is_zh(page: Path) -> bool:
    return page.name.endswith(ZH)


def violations(root: Path) -> list[str]:
    """One ``path[:line]: problem`` entry per violation under ``root``."""
    root = root.resolve()
    found, texts = [], {page.resolve(): page.read_text(encoding="utf-8") for page in pages(root)}
    for page, text in texts.items():
        name, other = page.relative_to(root).as_posix(), counterpart(page)
        if other not in texts:
            found.append(f"{name}: no {other.name}")
        elif not _is_zh(page):
            mine, theirs = parse(text), parse(texts[other])
            if [level for level, _ in mine[0]] != [level for level, _ in theirs[0]]:
                found.append(f"{name}: headings differ from {other.name}")
            if mine[1] != theirs[1]:
                found.append(f"{name}: {mine[1]} table rows, {other.name} {theirs[1]}")
            if [code(block) for block in mine[2]] != [code(block) for block in theirs[2]]:
                found.append(f"{name}: code differs from {other.name}")
        switch = False
        for number, _, _, target in links(text):
            dest, anchor = _resolve(page, target)
            if dest == other and number <= SWITCH_LINES:
                switch = True
                continue
            if not dest.exists():
                found.append(f"{name}:{number}: {target}: no such file")
            elif anchor and dest.suffix == ".md" and anchor not in anchors(texts.get(dest) or dest.read_text()):
                found.append(f"{name}:{number}: {target}: no such heading")
            elif dest in texts and _is_zh(dest) != _is_zh(page):
                found.append(f"{name}:{number}: {target}: links to the other language")
        if other in texts and not switch:
            found.append(f"{name}: no language switch to {other.name} in the first {SWITCH_LINES} lines")
    return found


def _switch(page: Path) -> str:
    other = counterpart(page).name
    return f"[English]({other}) | 简体中文" if _is_zh(page) else f"English | [简体中文]({other})"


def rewrite(root: Path) -> list[Path]:
    """Add language switches and point Chinese pages' links at Chinese pages (see the module docstring); returns the
    pages changed."""
    root, changed = root.resolve(), []
    for page in pages(root):
        page = page.resolve()
        other = counterpart(page)
        if not other.exists():
            continue
        text = page.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not any(_resolve(page, t)[0] == other and n <= SWITCH_LINES for n, _, _, t in links(text)):
            h1 = next((i for i, line in enumerate(lines[:SWITCH_LINES]) if line.startswith("# ")), -1)
            lines[h1 + 1:h1 + 1] = ([""] if h1 >= 0 else []) + [_switch(page)] + ([] if h1 >= 0 else [""])
        if _is_zh(page):
            edits = {}
            for number, start, end, target in links("\n".join(lines)):
                dest, anchor = _resolve(page, target)
                if target.startswith("#"):
                    source, translated, target = other, page, ""
                elif (dest.suffix == ".md" and not _is_zh(dest) and counterpart(dest).exists()
                      and not (dest == other and number <= SWITCH_LINES)):
                    source, translated, target = dest, counterpart(dest), target.partition("#")[0][:-3] + ZH
                else:
                    continue
                if anchor:
                    english_anchors = anchors(source.read_text(encoding="utf-8"))
                    chinese_anchors = anchors("\n".join(lines) if translated == page else translated.read_text())
                    if anchor in english_anchors and len(english_anchors) == len(chinese_anchors):
                        anchor = chinese_anchors[english_anchors.index(anchor)]
                    target += "#" + anchor
                edits.setdefault(number, []).append((start, end, target))
            for number, spans in edits.items():
                line = lines[number - 1]
                for start, end, target in sorted(spans, reverse=True):
                    line = line[:start] + target + line[end:]
                lines[number - 1] = line
        new = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        if new != text:
            page.write_text(new, encoding="utf-8")
            changed.append(page)
    return changed


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("check", "links"):
        print(__doc__.strip().splitlines()[-1])
        return 2
    root = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    if sys.argv[1] == "links":
        for page in rewrite(root):
            print(f"updated {page.relative_to(root.resolve()).as_posix()}")
        return 0
    found = violations(root)
    for entry in found:
        print(entry)
    print(f"docs_i18n: {len(found)} violation(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
