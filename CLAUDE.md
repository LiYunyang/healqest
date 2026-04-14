# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Architecture
### Core Package (`healqest/`)
- `ducc_sht.py`: For all partial-sky SHT operations.
- `weights.py`, `resp.py`, `qest.py`: main lensing reconstruction codes.
- `cinv/`: CMB inverse-variance filter (conjugate gradient solver).
- `healqest_utils.py`: general utilities.
- `startup.py`: interface between scripts and core library. The `Config` object handles all configuration and paths resolution.
### Scripts (`scripts/`,)
Main pipeline scripts (and some other small util scripts) that are official are stored in `scripts/`.

## Development Standards
**Core Philosophy**: Code is read 10x more than written. Optimize for readability and maintainability, not cleverness.

### Code Style
- Use `ruff` for formatting
- Configured in `pyproject.toml`
- Maximum line length: 110 characters
- Use `# fmt: off` sparingly for argument-heavy script lines where breaking hurts readability.
- Keep files small and focused
- Vertical formatting: related concepts close together, blank lines separate concepts
- Group related functions together

### Naming Conventions
- **Files**: underscored_lowercase (example_script.py)
- **Classes**: CamelCase (MyObject)
- **Functions/Variables**: snake_case (this_function)
- **Constants/Global variable**: UPPER_SNAKE_CASE (CONSTANT_VALUE)
- Use intention-revealing names that explain why something exists
- Avoid disinformation and meaningless distinctions (e.g., `data`, `info`, `manager`)
- Use pronounceable, searchable names
- Class names: nouns (e.g., `Map`, `Estimator`)
- Method names: verbs (e.g., `get_spec`, `make_map`)


## Documentation/Comments
- When asked to refactor the code, don't try to remove the existing docs unless explicitly asked to.
- Docstrings follows the numpy style, and the rules defined in `pyproject.toml`. No space between the parameter name and the comma, i.e "param1: str".
- Simple function requires simple docstring. No need to have per-parameter description for every function.
- Self-documenting code > comments > external docs
- Public APIs need clear documentation
- Include examples in documentation where helpful.
- Code should be self-explanatory - avoid comments when possible

## Object-Oriented Design Principles
- Small classes: measured by responsibilities, not lines
- Single Responsibility Principle: one reason to change
- High cohesion: class variables used by many methods
- Low coupling: minimal dependencies between classes
- Open/Closed Principle: open for extension, closed for modification

## Code Quality Principles
- **DRY (Don't Repeat Yourself)**: No duplication
- **YAGNI (You Aren't Gonna Need It)**: Don't build for hypothetical futures
- **KISS (Keep It Simple)**: Avoid unnecessary complexity
- **Boy Scout Rule**: Leave code cleaner than you found it

## Code Smells to Avoid
- Long functions or classes
- Duplicate code
- Dead code (unused variables, functions, parameters)
- Feature envy (method more interested in other class)
- Inappropriate intimacy (classes knowing too much about each other)
- Long parameter lists
- Primitive obsession (overusing primitives instead of small objects)
- Switch/case statements (consider polymorphism)

## Error Handling
- Use exceptions, not return codes or error flags
- Fail fast: raise exceptions as soon as an error condition is detected.
- Write `try-except` only when the exception is expected and can be handled meaningfully.
- Provide context in exception messages

## Testing
- Unit tests live in `test/` and configured by `pyproject.toml`.
- Fast, Independent, Repeatable, Self-validating, Timely (F.I.R.S.T.)
- One assert per test (or one concept)
- Test code quality equals production code quality
- Readable test names that describe what's being tested
- Arrange-Act-Assert pattern
