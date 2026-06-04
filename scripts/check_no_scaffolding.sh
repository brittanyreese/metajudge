#!/usr/bin/env bash
# Pre-commit guard: block agent/assistant scaffolding and secrets from commits.
# Keeps the public, peer-review-facing history clean. The shareable context is
# AGENTS.md; private machine context stays in the gitignored CLAUDE.md.
set -euo pipefail

patterns=(
  '^\.claude/'
  '^\.serena/'


  '^CLAUDE\.md$'

  '(^|/)\.env($|\.)'
  '(^|/)\.envrc$'
  '\.pem$'
  '(^|/)id_rsa'
  '(^|/)secrets?\.'
)

staged="$(git diff --cached --name-only --diff-filter=A)"
[ -z "$staged" ] && exit 0

blocked=""
while IFS= read -r file; do
  for p in "${patterns[@]}"; do
    if printf '%s\n' "$file" | grep -qE "$p"; then
      blocked="${blocked}  ${file}\n"
    fi
  done
done <<< "$staged"

if [ -n "$blocked" ]; then
  printf '\nBLOCKED: scaffolding or secrets cannot be committed:\n' >&2
  printf '%b' "$blocked" >&2
  printf 'These paths are gitignored on purpose. Shareable context is AGENTS.md.\n' >&2
  printf 'If this is intentional, edit scripts/check_no_scaffolding.sh.\n\n' >&2
  exit 1
fi

exit 0
