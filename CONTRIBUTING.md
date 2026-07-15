# Contributing to Tri-Calib

Thank you for your interest in contributing to Tri-Calib!
Whether you are reporting a bug, suggesting a feature, or
submitting a fix, your contribution is welcome.

---

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Development Setup](#development-setup)
- [Running the Tests](#running-the-tests)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Code Style](#code-style)
- [Questions](#questions)

---

## Reporting Bugs

Please open a [GitHub Issue](https://github.com/dfki-av/Tricalib/issues)
and include the following information:

- **Operating system** (Windows 10/11, Ubuntu 22.04, macOS, etc.)
- **Python version** (`python --version`)
- **Tri-Calib version or commit hash** (`git rev-parse --short HEAD`)
- **Steps to reproduce** — what data did you load, what did you click,
  and what happened?
- **Expected behaviour** — what should have happened?
- **Error message or screenshot** — copy the full traceback from the
  terminal if applicable

The more detail you provide, the faster the issue can be resolved.

---

## Suggesting Features

Open a [GitHub Issue](https://github.com/dfki-av/Tricalib/issues) with
the label `enhancement` and describe:

- **The use case** — what problem are you trying to solve?
- **The proposed solution** — what would you like the tool to do?
- **Alternatives you considered** — are there workarounds?

We are particularly interested in contributions related to:
- Support for additional sensor modalities (e.g. RADAR, thermal cameras)
- Semi-automatic correspondence detection across modalities
- Additional export formats for calibration results
- Improvements to the event frame generation pipeline

---

## Development Setup

**1. Clone the repository:**
```bash
git clone https://github.com/dfki-av/Tricalib.git
cd Tricalib
```

**2. Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

**3. Install in editable mode with all dependencies:**
```bash
pip install -e .
pip install -r requirements.txt
```

**4. Verify the installation:**
```bash
python -m tricalib
```

The GUI should launch without errors.

---

## Running the Tests

Tri-Calib uses [pytest](https://pytest.org) for testing.
Run the full test suite from the repository root with:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_optimizer.py
pytest tests/test_projection.py
pytest tests/test_io.py
```

To run with verbose output showing each test name:

```bash
pytest -v
```

Please ensure all tests pass before submitting a pull request.
If you add new functionality, include corresponding tests in the
`tests/` directory following the existing naming convention
(`test_<module>.py`).

---

## Submitting a Pull Request

We use a **feature branch workflow**:

**1. Fork the repository** and create a branch from `main`:
```bash
git checkout -b feature/your-descriptive-branch-name
```

Use a descriptive branch name that reflects the change:
- `feature/radar-support` for new features
- `fix/event-window-crash` for bug fixes
- `docs/improve-readme` for documentation updates

**2. Make your changes** and commit with a clear message:
```bash
git commit -m "Add radar sensor calibration pathway"
```

Keep commits focused — one logical change per commit.

**3. Run the tests** and ensure they all pass:
```bash
pytest
```

**4. Open a pull request** against the `main` branch and describe:
- What the change does
- Why it is needed
- How you tested it
- Any known limitations or follow-up work

---

## Code Style

Tri-Calib uses [autopep8](https://github.com/hhatto/autopep8) for
code formatting. Please format your code before committing:

```bash
pip install autopep8
autopep8 --in-place --aggressive --recursive .
```

Additional conventions:
- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines
- Use NumPy-style docstrings for all public functions and classes
- Add type hints to new functions where possible

---

## Questions

If you have a question not answered by the
[documentation](https://dfki-av.github.io/Tricalib/) or this guide:

- **Open a GitHub Issue** with the label `question`
- **Email:** rahul.jakkamsetty@dfki.de

---

## Acknowledgement

Contributors will be acknowledged in the project documentation.
Significant contributions may be recognised in future publications
that build on this work.