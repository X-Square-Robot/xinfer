# Contributing to XInfer

XInfer is in private development. Read [docs/design.md](docs/design.md),
including its implementation-status section, before proposing implementation
changes.

The GitHub distribution ships only Wall-OSS. When syncing from upstream, import reviewed changes for the framework and this model.

## Scope and architecture

- Keep model semantics, runtime scheduling, hardware capabilities, and kernel
  implementations behind their documented interfaces.
- Implement the colocated GPU layout first. GPU-stage split execution is
  deferred; CPU frontend processes remain in scope.
- Use English for code and comments. Documentation pages exist in English and
  Simplified Chinese (see "Repository contents").
- Add maintained code when it has an implementation or validation purpose;
  avoid placeholder modules that imply unsupported functionality.

## Merge requests

- **One concern per merge request.** Keep refactoring, renaming and
  formatting-only changes in their own merge request and say "no functional
  change" in its title. Do not include unrelated edits.
- **Size.** Aim for about 400 changed lines, not counting tests and fixtures.
  Split anything above about 1000 lines into stacked merge requests, or agree
  on the design in an issue first.
- **Description.** Use the default template: problem, changes, validation,
  limitations and references. List the checks actually run with their results,
  and separate measured results from hypotheses and targets. Change-by-change
  measurements, comparisons with earlier revisions and where the evidence is
  stored belong here, not in the repository.
- **Commits.** Imperative subject without issue numbers; the body explains why.
  Squash fix-up commits before review. Put `Closes #N` only in the
  description's references: GitLab closes the issue when the merge commit,
  which carries the description, lands.
- **Review.** Merge only with CI passing and the branch up to date with `main`.
  Keep a merge request in draft until its validation is complete. Answer every
  review comment with a change or an explanation.
- **AI assistance.** Agent-written merge requests follow the same rules. The
  responsible person reviews every changed line, and the description says
  which parts an agent wrote.

## Code

- Write what this change needs: reuse existing code, and do not add
  abstractions, options or flags for hypothetical future use.
- No debug output, commented-out code or dead code.
- Comments explain why. No issue or merge-request numbers, dates or names in
  code or comments.
- Experiment-only scripts do not go into `xinfer/` or `tools/`; keep them in
  ignored `artifacts/` or outside the repository.
- Document numerical behavior, ownership, and lifecycle implications when
  changing execution or batching. Do not change precision thresholds merely
  to make an optimization pass.

## Tests

- A bug fix comes with a regression test, and the description states that the
  test fails without the fix.
- New behavior comes with tests; prefer tests that run on the CPU.
- No tests that depend on timing or on random collisions to expose a problem.
- Test fixtures are small and have an identified consumer.

## Validation

GitLab CI (`.gitlab-ci.yml`) runs on every merge request and on `main`:
`ruff check .` (settings in `pyproject.toml`), the CPU test suite,
`tools/check_repo.py` (the repository content rules below) and
`tools/docs_i18n.py check` (the documentation languages, below). GPU and asset
tests skip in CI; run them on a GPU machine and report them in the merge
request. Do not report planned checks as passing. Each implementation change
should document runnable checks appropriate to the functionality it adds.

As the implementation becomes available, validation should cover:

- CPU contracts for admission, stage progression, cancellation, resource
  ownership, configuration, and dependency direction.
- Backend conformance against reference operators, including graph capture and
  the declared determinism and batch invariance.
- Model alignment with fixed inputs, initial noise, and reference outputs.
- Same-configuration output invariance across supported runtime paths.
- Repeated latency and fleet-capacity measurements for each supported platform.

Performance reports must identify the source revision, hardware/software
environment, model and precision configuration, inputs, timing boundaries,
workload, repetitions, numerical checks, and evidence location. Preserve
negative results and known limitations.

## Repository contents

Commit source, maintained tools, documentation, and small deterministic test
fixtures with an identified consumer. Keep checkpoints, datasets, raw request
logs, traces, profiler captures, and bulk experiment output in ignored
`artifacts/` or controlled external storage. Commit concise results summaries
and evidence references instead of raw experiment archives.

Documentation describes the current state: usage, design and validated
results, each result with its configuration and its evidence archive (name and
sha256). It has no experiment logs, history pages or comparisons with earlier
revisions; update it in the same merge request as any user-visible change.

Every documentation page (`README.md`, `docs/**/*.md`) exists as an English page
`X.md` and a Simplified Chinese page `X.zh-CN.md`, which link to each other in
their first lines. A merge request that changes a page changes both, with the
same headings, tables and code (comments, diagrams and other `text` blocks are
translated), and links in a Chinese page lead to Chinese pages. Write a Chinese
page with the English link targets, then run `python tools/docs_i18n.py links`
to point them at the Chinese pages and their headings. CI fails on a missing
counterpart or language switch, a structural difference within a pair, a
relative link to a missing file or heading, and a link into the other language.
Use these terms consistently:

| English | 中文 | English | 中文 |
|---|---|---|---|
| bitwise equal | 逐位一致 | batching | 组批 |
| precision gate | 精度门禁 | batch-invariant | 批次不变 |
| lane | 通道（lane） | capacity | 容量 |
| stage | 阶段 | missed slot | 漏槽 |
| handoff | 交接 | open-loop | 开环 |
| warmup | 预热 | evidence archive | 证据包 |
| admission | 准入 | production service | 生产服务 |

`tools/check_repo.py` fails CI on documentation history pages, machine-specific
paths or machine names, files over 1 MiB (unless listed in the script), and
issue or merge-request numbers in Python code.

Do not commit credentials or machine-specific private configuration. Preserve
source provenance when porting code. Licensing and external publication remain
pending and must be resolved before public release.
