"""Compatibility shim: the service now lives in the `evidence` package.

`python app.py` and `python -m evidence` both serve it.
"""

from evidence.__main__ import main

if __name__ == "__main__":
    main()
