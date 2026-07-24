#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage:
  read-ledger-sections.sh --index FILE
  read-ledger-sections.sh FILE REGEX [MAX_SECTIONS=3] [MAX_LINES=220]

Prints matching Markdown H2 sections only. Matching is case-insensitive.
USAGE
    exit 2
}

[[ $# -ge 2 ]] || usage

if [[ "$1" == "--index" ]]; then
    file="$2"
    [[ -f "$file" ]] || { printf 'missing file: %s\n' "$file" >&2; exit 1; }
    grep -nE '^#{1,3} ' "$file" || true
    exit 0
fi

file="$1"
regex="$2"
max_sections="${3:-3}"
max_lines="${4:-220}"
[[ -f "$file" ]] || { printf 'missing file: %s\n' "$file" >&2; exit 1; }
[[ "$max_sections" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$max_lines" =~ ^[1-9][0-9]*$ ]] || usage

awk -v pat="$regex" -v maxs="$max_sections" -v maxl="$max_lines" '
function flush(  i) {
    if (!in_section || !matched || printed_sections >= maxs) return
    printed_sections++
    for (i = 1; i <= n && printed_lines < maxl; i++) {
        print buf[i]
        printed_lines++
    }
    if (printed_lines < maxl) { print ""; printed_lines++ }
}
BEGIN { pat=tolower(pat); in_section=0; matched=0; n=0; printed_sections=0; printed_lines=0 }
/^## / {
    flush()
    if (printed_sections >= maxs || printed_lines >= maxl) exit
    delete buf; n=0; in_section=1; matched=(tolower($0) ~ pat); buf[++n]=$0; next
}
{
    if (in_section) {
        buf[++n]=$0
        if (tolower($0) ~ pat) matched=1
    }
}
END { if (printed_sections < maxs && printed_lines < maxl) flush() }
' "$file"
