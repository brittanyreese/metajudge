#!/usr/bin/env bash
# Pre-commit guard: keep the public, peer-review-facing history clean.
# Shareable project conventions live in AGENTS.md; machine-local tool
# context stays gitignored.
#
# Design: allowlist-first, fail-safe. Local tooling scaffolding always lands
# as NEW top-level directories. A denylist only blocks the tools we already
# know about; a
# top-level allowlist blocks every newly-added top-level entry by default,
# including tools that do not exist yet. New nested files under an approved
# root (src/, tests/, docs/, ...) flow through freely. Secrets, which can be
# nested anywhere, stay on an explicit position-independent denylist.
set -euo pipefail

# Approved top-level entries. A newly-added file whose first path component is
# not in this set is blocked. To add a genuinely new top-level file or dir,
# add it here in the same commit.
allowed_toplevel=(
  ".env.example"
  ".github"
  ".gitignore"
  ".mailmap"
  ".pre-commit-config.yaml"
  "AGENTS.md"
  "CHANGELOG.md"
  "CITATION.cff"
  "CONTRIBUTING.md"
  "LICENSE"
  "README.md"
  "SPEC.md"
  "docs"
  "examples"
  "pyproject.toml"
  "scripts"
  "sim"
  "src"
  "tests"
  "uv.lock"
)

# Explicit exceptions to the secret denylist below -- exact filenames that are
# templates, never real values. Checked first so the broad `.env*` pattern
# below doesn't sweep up the conventional template file.
secret_safe_literals=(
  '(^|/)\.env\.example$'
)

# Secrets and key material can appear nested anywhere, so they stay on an
# explicit denylist matched against the full path.
secret_patterns=(
  '(^|/)\.env($|\.)'
  '(^|/)\.envrc$'
  '\.pem$'
  '(^|/)id_rsa'
  '(^|/)id_ed25519'
  '(^|/)secrets?\.'
  '\.p12$'
  '\.pfx$'
)

# Private subtrees nested under an otherwise-allowed top-level root. The
# top-level allowlist permits docs/, but a private build-plan subtree under it
# is denied; its sanitized public mirror is docs/roadmap.md.
nested_deny_patterns=(
  '^docs/superpowers/'
)

staged="$(git diff --cached --name-only --diff-filter=A)"
[ -z "$staged" ] && exit 0

blocked=""
while IFS= read -r file; do
  [ -z "$file" ] && continue

  # Safe-template exceptions (checked before the secret denylist).
  is_safe=""
  for p in "${secret_safe_literals[@]}"; do
    if printf '%s\n' "$file" | grep -qE "$p"; then
      is_safe="yes"
      break
    fi
  done

  # Secret denylist (matched anywhere in the path).
  if [ -z "$is_safe" ]; then
    for p in "${secret_patterns[@]}"; do
      if printf '%s\n' "$file" | grep -qE "$p"; then
        blocked="${blocked}  ${file}  (secret/key material)\n"
        continue 2
      fi
    done
  fi

  # Private-subtree denylist (nested under an allowed root).
  for p in "${nested_deny_patterns[@]}"; do
    if printf '%s\n' "$file" | grep -qE "$p"; then
      blocked="${blocked}  ${file}  (private subtree)\n"
      continue 2
    fi
  done

  # Top-level allowlist.
  top="${file%%/*}"
  permitted=""
  for a in "${allowed_toplevel[@]}"; do
    if [ "$top" = "$a" ]; then
      permitted="yes"
      break
    fi
  done
  if [ -z "$permitted" ]; then
    blocked="${blocked}  ${file}  (top-level '${top}' not in allowlist)\n"
  fi
done <<< "$staged"

if [ -n "$blocked" ]; then
  printf '\nBLOCKED: these new paths are not permitted in the public history:\n' >&2
  printf '%b' "$blocked" >&2
  printf '\nScaffolding and secrets are gitignored on purpose; AGENTS.md is the\n' >&2
  printf 'shareable context. If a path is a legitimate new project file, add its\n' >&2
  printf 'top-level entry to allowed_toplevel in scripts/check_no_scaffolding.sh\n' >&2
  printf 'in this same commit.\n\n' >&2
  exit 1
fi

exit 0
