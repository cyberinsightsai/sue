# AGENTS.md - Guidelines for Agentic Coding in SUE Repository

## Build/Lint/Test Commands

- **Install dependencies**: `pip install -r requirements.txt`
- **Run application**: `streamlit run raspberry_pi_rag.py`
- **Lint code**: `black . && flake8 .` (format with Black, check with Flake8)
- **Type check**: `mypy .` (if mypy installed)
- **Run tests**: `pytest` (no existing tests; add pytest to requirements for new tests)
- **Run single test**: `pytest path/to/test_file.py::test_function` (once tests are added)

## Code Style Guidelines

- **Imports**: Group as standard library, third-party, local. Use absolute imports.
- **Formatting**: Follow PEP8; use Black for auto-formatting (line length 88).
- **Types**: Use type hints from `typing` module for all functions and methods.
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants.
- **Error handling**: Use try-except blocks; log errors with `st.error()` for Streamlit UI.
- **Docstrings**: Add docstrings to all classes and public methods using triple quotes.
- **Comments**: Add comments for complex logic; avoid unnecessary comments.
- **Warnings**: Suppress non-critical warnings with `warnings.filterwarnings("ignore")` if needed.
- **Security**: No hardcoded secrets; use environment variables or config files.

No Cursor rules or Copilot instructions present in this repository.