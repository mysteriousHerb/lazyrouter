# Using LazyRouter with uv

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver, written in Rust. It's significantly faster than pip and provides better dependency resolution.

## Why uv?

- **10-100x faster** than pip for installing packages
- **Better dependency resolution** - handles conflicts more intelligently
- **Built-in virtual environment management**
- **Compatible with pip** - uses the same package index (PyPI)

## Installation with uv

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 2. Set up LazyRouter

```bash
# Clone the repository
git clone https://github.com/yourusername/lazyrouter.git
cd lazyrouter

# Create virtual environment and install dependencies
# This reads pyproject.toml and creates .venv automatically
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run LazyRouter

```bash
# Run with uv (automatically uses .venv)
uv run python main.py

# Or activate the virtual environment manually
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Then run normally
python main.py
```

## Common uv Commands

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add package-name

# Remove a dependency
uv remove package-name

# Update dependencies
uv sync --upgrade

# Run a script
uv run python script.py

# Run tests
uv run pytest

# Install dev dependencies
uv sync --all-extras
```

## Project Structure with uv

LazyRouter uses `pyproject.toml` for dependency management:

```toml
[project]
name = "lazyrouter"
version = "0.1.0"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    # ... other dependencies
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.3",
    "black>=24.10.0",
    # ... dev dependencies
]
```

## Comparison: uv vs pip

| Task | uv | pip |
|------|-----|-----|
| Install all deps | `uv sync` | `pip install -r requirements.txt` |
| Add dependency | `uv add package` | `pip install package` + manual edit |
| Create venv | `uv sync` (automatic) | `python -m venv venv` |
| Run script | `uv run python script.py` | `python script.py` |
| Speed | ⚡ Very fast | 🐌 Slower |

## Troubleshooting

### "uv: command not found"
- Install uv following the instructions above
- Make sure uv is in your PATH

### "No such file or directory: pyproject.toml"
- Make sure you're in the lazyrouter directory
- The pyproject.toml file should be in the project root

### Virtual environment issues
- Delete `.venv` folder and run `uv sync` again
- Make sure you're using Python 3.10 or higher

## Migration from pip

If you're already using pip with `requirements.txt`, you can continue using it. Both work fine:

```bash
# Using uv (recommended)
uv sync

# Using pip (still works)
pip install -r requirements.txt
```

The `requirements.txt` file is kept for backwards compatibility, but `pyproject.toml` is the source of truth.

## Learn More

- [uv Documentation](https://docs.astral.sh/uv/)
- [uv GitHub Repository](https://github.com/astral-sh/uv)
- [Why uv is fast](https://astral.sh/blog/uv)
