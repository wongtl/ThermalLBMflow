<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Contributing to ThermalLBMflow

Thanks for taking the time to contribute.

This repository uses the standard GitHub workflow:

- open an issue for bugs, regressions, documentation problems, or feature ideas
- submit changes as a pull request against this repository
- keep pull requests focused so review stays clear and manageable

## Before You Open a Pull Request

Please:

- describe the problem and proposed fix clearly
- mention any CPU/GPU parity implications when behavior differs by backend
- include validation details such as local build, smoke-run, or relevant test output
- update documentation when user-facing behavior changes

## Scope and Upstream Context

ThermalLBMflow is built on top of waLBerla and keeps the waLBerla source tree
in this repository so the app can build from a single checkout.

If your change is specific to ThermalLBMflow, open the issue or pull request
here.

If your change is a generally useful waLBerla core improvement, it is still fine
to start here, but please call that out explicitly in the discussion so we can
decide whether it should also be proposed upstream.

## Coding Expectations

When possible:

- keep patches minimal and well-scoped
- preserve CPU/GPU behavior alignment unless divergence is intentional and documented
- preserve checkpoint/restart correctness
- keep startup and runtime logging traceable
- avoid bundling unrelated refactors with functional changes

## License

ThermalLBMflow is distributed under the GNU General Public License v3 or later.
By contributing, you agree that your contribution may be distributed under that
license. See [COPYING.txt](COPYING.txt).
