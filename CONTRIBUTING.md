# Contributing

This is a two-person student project. Keep changes focused, explainable, and easy for your teammate to review.

## Branches and pull requests

- Keep `main` stable; do not develop directly on it.
- Create a descriptive branch from the latest `main`, for example `feature/phase1-data-pipeline`, `feature/frequency-branch`, or `docs/roadmap`.
- Before starting, inspect `git status`, fetch and review the latest remote changes, and preserve any local work.
- Make small commits with concise messages describing the change.
- Open a pull request for substantial work. Summarize the goal, changed files, validation performed, and any known limitations.
- Review conflicts carefully. Do not force-push, rewrite shared history, or discard another contributor's changes.

## Shared interfaces

Agree on dataset labels, split rules, tensor/image conventions, and model input/output contracts before implementing modules in parallel. Record those decisions in the documentation or configuration so both contributors use the same assumptions.

## Data and generated files

Do not commit datasets, credentials, local environment files, virtual environments, model checkpoints, or generated outputs. Check `git status` and the complete diff before committing or opening a pull request.

## Validation

Run the validation relevant to the change and report exactly what was run and its result. Do not report unrun checks as passing. Add tests alongside implemented behavior in the phase where that behavior is introduced.
