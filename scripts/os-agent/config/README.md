# Harness configuration

`parser-tests.tsv` is the parser-test registry and selection authority.
Each non-comment row contains five tab-separated fields:

1. logical name;
2. self-contained test path;
3. `module:<path>` or `fixture:<path>` binding;
4. semicolon-separated changed-path globs;
5. informational tags.

Quick gates run only entries selected by parser/test/runner path globs. Registry edits are checked structurally and by the isolated Harness self-test; `--full` runs every registered parser test.
A new or changed `tests/*parser*.py` file that is not registered is `UNRESOLVED`.

`parser-waivers.tsv` is deliberately empty. Temporary waivers must bind the exact
baseline, test bytes, binding bytes, normalized failure signature and expiration.
No project stage or historical HEAD is hard-coded into Harness source.
