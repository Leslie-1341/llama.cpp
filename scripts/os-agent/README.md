# OS Agent Harness v2

Harness v2 keeps the fail-closed verdict model while reducing repeated work and terminal log volume.

## Default profiles

| Mode | Profile | Behavior |
|---|---|---|
| `implement` | `quick` | Changed-file syntax, verified target mapping/build, selected parser tests, relevant Skill/ledger checks |
| `review` | `verify` | Reuses a compatible unchanged PASS artifact; otherwise quick checks plus changed-TU clang-tidy |
| `review-fix` | `quick` | Revalidates the current surface and reuses unchanged build/parser components |
| `audit` | `audit` | Read-only source/worktree checks and validation plan; no configure, build, parser execution or self-test |

Use `--full` only at a stable node. It runs all registered parser tests and full isolated Harness tests. It does not run model experiments.

## Compact output

`--quiet` emits one `OS_AGENT_GATE_RESULT` line. Full command output is retained under the artifact `logs/` directory. Normal mode prints one line per check and a compact failure excerpt. `--verbose` streams full logs.

## Evidence reuse

The global fingerprint binds:

- HEAD and complete tracked worktree diff;
- untracked file bytes;
- build directory identity;
- `CMakeCache.txt` and `compile_commands.json` hashes;
- Harness version and tool versions.

`review` reuses an unchanged PASS artifact. Build, parser and clang-tidy checks also have component fingerprints, so a parser-only `review-fix` does not rebuild unchanged C++ targets.

Use `--fresh` to disable all reuse.

## Parser registry

Parser tests are configured in `config/parser-tests.tsv`. Quick gates select tests by changed parser/test/runner globs; `--full` runs all. Temporary waivers belong in `config/parser-waivers.tsv` and must bind exact file and failure hashes plus an expiration time. The default waiver file is empty.

## Self-test

```bash
bash scripts/os-agent/tests/test-harness.sh --fast
bash scripts/os-agent/tests/test-harness.sh --full
```

Both suites use temporary Git/CMake fixtures. They never edit the caller's real worktree.
