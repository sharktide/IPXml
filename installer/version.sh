#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION_FILE="$REPO_ROOT/VERSION"

if [[ ! -f "$VERSION_FILE" ]]; then
  echo "ERROR: VERSION file not found: $VERSION_FILE" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 get | $0 set <new-version>" >&2
  exit 1
fi

action="$1"
shift

function get_version() {
  cat "$VERSION_FILE" | tr -d '\r' | sed 's/^\s*//;s/\s*$//'
}

function set_version() {
  if [[ $# -ne 1 ]]; then
    echo "Usage: $0 set <new-version>" >&2
    exit 1
  fi
  local new_version="$1"
  printf '%s\n' "$new_version" > "$VERSION_FILE"

  python3 - "$REPO_ROOT" "$new_version" <<'PY'
import pathlib
import re
import sys
root = pathlib.Path(sys.argv[1])
version = sys.argv[2]
section = None
pattern = re.compile(r'^\s*\[(.+?)\]\s*$')
for toml_path in sorted(root.joinpath('crates').rglob('Cargo.toml')):
    text = toml_path.read_text()
    lines = text.splitlines(keepends=True)
    modified = False
    current_section = None
    for idx, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            current_section = m.group(1)
        if current_section in ('package', 'package.metadata.bundle') and re.match(r'^\s*version\s*=\s*".*"\s*$', line):
            lines[idx] = f'version = "{version}"\n'
            modified = True
    if modified:
        toml_path.write_text(''.join(lines))
PY
}

case "$action" in
  get)
    get_version
    ;;
  set)
    set_version "$@"
    ;;
  *)
    echo "Usage: $0 get | $0 set <new-version>" >&2
    exit 1
    ;;
esac
