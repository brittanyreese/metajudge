# Contributing

A solo, open-source research project. These conventions keep the history clean enough for peer review and for others to build on. The canonical working context is [AGENTS.md](AGENTS.md).

## Setup

```bash
uv sync   # runtime + dev deps
uv run pre-commit install   # once: lint/format/guard run on every commit
```

## Quality gates

Every change that lands on `main` must pass:

```bash
uv run ruff check .          # lint
uv run ruff format --check . # format
uv run pyright               # type-check (strict)
uv run pytest                # tests (coverage runs automatically)
```

pre-commit runs these locally; CI runs them on push. Numerical correctness is non-compressible (SPEC R19): every statistic is tested against a pinned external reference (the `krippendorff` package, R `irr`/`TAM`, statsmodels), not internal consistency. A wrong alpha or DIF is a wrong answer, so the reference fixtures are part of the contract.

## Commits

- Conventional Commits: `type(scope): subject` (subject <=72 chars; body explains the why).
- One logical change per commit. Keep formatting-only changes in their own `style:` commit; never bundle a repo-wide reformat into a feature commit.
- New runtime dependency: justify it in the commit. Runtime deps are limited to `numpy`, `pandas`, `scipy`, `krippendorff`; `pingouin`/`statsmodels` are dev/test oracles only.

## Branching and releases

- Trunk-based: `main` stays green and releasable.
- Small changes go straight to `main`. Larger work uses a short-lived branch merged back promptly, so `main` never goes stale.
- Tag releases `vX.Y.Z`. For any reported result, cite the exact tagged commit that produced it.

## Pull requests

`main` is protected: changes land via a pull request with CI green (the 3.11/3.12/3.13 matrix) and a linear history. The sole maintainer can bypass for genuine trivia, but the default path is a PR: it gives every change a CI gate and a self-review pass.

Flow for non-trivial work:

1. Branch off `main`.
2. Build test-first (the RED target is a pinned external reference value); keep commits clean and conventional.
3. Self-review the diff before opening (`/code-review`).
4. `gh pr create`; CI runs on the branch.
5. Squash-merge once green. Tag if it is a release.

## Hygiene

- Never commit agent scaffolding or secrets. The pre-commit guard (`scripts/check_no_scaffolding.sh`) blocks them. The shareable context is `AGENTS.md`; the private machine context stays in the gitignored `CLAUDE.md`.
- Get history clean before the first push: rewriting public history needs a force-push and is disruptive to anyone who has cloned.

## License

MIT (see [LICENSE](LICENSE)). Contributions are accepted under the same license.
