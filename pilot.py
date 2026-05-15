"""Backwards-compatibility shim. Prefer the `psca` console script after `pip install -e .`."""
from psca.cli import main

if __name__ == "__main__":
    main()
