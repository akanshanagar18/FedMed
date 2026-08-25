# Git Workflow

## Author
[Author Placeholder]

## Purpose
Defines the branching strategy and release process for FedMed to maintain a stable, production-ready codebase.

## Branching Model
FedMed follows a Gitflow-inspired model:
- `main`: Reflects the production-ready state. Commits here must be tagged releases.
- `development`: The primary integration branch. All feature branches merge here.
- `feature/*`: Branched from `development`. Used for new features (e.g., `feature/secure-aggregation`).
- `bugfix/*`: Branched from `development`. Used to fix non-critical bugs.
- `hotfix/*`: Branched directly from `main`. Used to fix critical production issues, then merged into both `main` and `development`.

## Commit Standards
We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` A new feature.
- `fix:` A bug fix.
- `docs:` Documentation only changes.
- `refactor:` Code changes that neither fix a bug nor add a feature.
- `chore:` Changes to the build process or auxiliary tools.

## Pull Request Process
1. Ensure your branch is up to date with `development`.
2. Run `make test` and `make lint` locally.
3. Open a PR against `development` using the standard PR template.
4. Wait for code review and address all feedback before merging.
