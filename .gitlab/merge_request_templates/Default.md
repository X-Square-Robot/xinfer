## Problem

Describe the concrete problem or missing capability.

## Changes

Describe the resulting behavior and relevant interface or lifecycle changes.

## Validation

List the commands or checks actually run and their results. For performance
changes, include the configuration, timing boundary, numerical checks,
repetitions, and evidence location.

## Limitations

State untested cases, known risks, and follow-up work. Distinguish planned
validation from completed validation.

## References

Link the relevant design sections, issues, or concise evidence reports.

## Checklist

- [ ] One concern; refactoring or formatting-only changes are in their own merge request.
- [ ] The code is minimal: no unused abstractions, debug output, commented-out code, or issue numbers in code.
- [ ] New behavior has tests; a bug fix has a regression test that fails without the fix.
- [ ] User-visible changes are documented in the English and Chinese pages; the docs describe the current state (no experiment logs, history or machine paths).
- [ ] Measurements and comparisons are in this description, with their configuration and evidence archive.
- [ ] CI passes; GPU checks that CI cannot run are reported above.
- [ ] Agent-written parts are stated, and every changed line has been reviewed.
