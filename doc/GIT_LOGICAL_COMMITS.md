# Logical Commit Workflow

This repository enforces commit message quality via local git hooks.

## Enable Once Per Clone

```bash
cd /workspaces/walberla-dev/ThermalLBMflow
bash utilities/git/setup-logical-commits.sh
```

## What Is Enforced

- `commit-msg` hook:
  - Subject length must be `<= 72` characters.
  - Subject must use `<area>: <meaningful change>`.
  - Generic subjects like `wip`, `update`, `misc` are rejected.

## Commit Template

The setup script configures `.gitmessage-logical-unit.txt`:

```text
<area>: <meaningful change>

Why:
- <problem or motivation>

What:
- <key implementation detail>

Validation:
- <command and result>
```
