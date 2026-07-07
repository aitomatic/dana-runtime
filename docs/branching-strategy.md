# Branching Strategy

Dana Runtime uses a **simplified two-branch model**: one integration branch and one release branch.

```
feature/* ──PR──> develop ──PR──> master
                                  |
                     hotfix/* ────┘ (branched from master; merged back to master + develop)
```

## Long-lived Branches

| Branch    | Purpose                                                               | Accepts PRs from     |
|-----------|----------------------------------------------------------------------|----------------------|
| `develop` | Integration branch. All feature work merges here first. **Default branch.** | `feature/*`          |
| `master`  | Release branch. Every merge produces a version tag + GitHub Release.  | `develop`, `hotfix/*` |

## Short-lived Branches

### Feature branches (`feature/*`)

- Branch from `develop`
- Merge back into `develop` via PR
- Naming: `feature/<descriptive-slug>` (e.g. `feature/timeline-compression`)
- Delete after merge

### Hotfix branches (`hotfix/*`)

- Branch from `master` for critical production bugs
- Merge into **both** `master` and `develop` (keep develop in sync)
- Bump the version on the hotfix branch manually before merging into `master`
- Naming: `hotfix/<version-or-slug>` (e.g. `hotfix/fix-crash`)
- Delete after merge

## Release Flow

1. `develop` accumulates features via merged feature branches
2. When ready for release, set the target version in `pyproject.toml` on `develop`
3. Open a PR `develop → master`. CI auto-bumps the patch version if `develop` is not already ahead
4. Merge the PR → [`release-on-merge-to-master`](../.github/workflows/release-on-merge-to-master.yml) tags `v<version>` and creates a GitHub Release

## Versioning

- Source of truth: `version` in [`pyproject.toml`](../pyproject.toml)
- To ship a **minor/major** bump (e.g. `0.2.0`, `1.0.0`), set it on `develop` before opening the release PR — the auto-bumper detects `develop` is already ahead and skips
- Otherwise the auto-bumper raises the patch component (`0.1.3` → `0.1.4`)
- Tags follow `v<version>` (e.g. `v0.2.0`)

## Branch Protection

Enforced by [`.github/workflows/branch-policy.yml`](../.github/workflows/branch-policy.yml) plus repository rulesets:

- **`master`** only accepts PRs from `develop` or `hotfix/*`
- Direct pushes to `master` are blocked (no non-fast-forward, no deletion); merges require review + signatures
- Any PR violating the source-branch rule is rejected by the `check-branch-policy` job

## Quick Reference

```text
feature/* ──PR──> develop ──PR──> master ──> tag vX.Y.Z + GitHub Release
                     ^                |
                     |     hotfix/* ──┘  (also merged back to develop)
                     +-------------------+
```
