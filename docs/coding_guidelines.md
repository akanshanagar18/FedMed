# Coding Guidelines

## Author
[Author Placeholder]

## Purpose
Establishes the code quality and architectural standards required for all contributions to FedMed.

## Architectural Principles
1. **Separation of Concerns:** Deep learning code must not mix with FL orchestration code.
2. **SOLID Design:** Modules should have a single responsibility and be open for extension but closed for modification.
3. **No Magic Strings:** All constants must reside in `configs/constants.py`.

## Python Standards
- **Formatting:** Code must be formatted using `black` (line length 88) and `isort`.
- **Linting:** Code must pass `flake8` checks without warnings.
- **Typing:** Strict type hints are mandatory. Code must pass `mypy`.
- **Documentation:** Every module, class, and public function must have a clear docstring explaining its purpose, arguments, and return types.

## Pull Request Requirements
- Code must include unit tests.
- CI pipelines must pass before merging.
- Requires approval from at least one core maintainer.
