"""Offline dependency check. Never installs packages or accesses the network."""

import importlib
import platform
import re
import struct
import sys

MINIMUM = {
    "numpy": (1, 26),
    "scipy": (1, 13),
    "pandas": (2, 2),
    "openpyxl": (3, 1),
    "torch": (2, 2),
    "fastapi": (0, 115),
    "pydantic": (2, 0),
    "uvicorn": (0, 30),
}

PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "openpyxl",
    "torch",
    "fastapi",
    "pydantic",
    "uvicorn",
)


def check():
    print(
        "Python: {} | {} bit | {}".format(
            platform.python_version(), 8 * struct.calcsize("P"), sys.executable
        )
    )
    if sys.version_info[:2] != (3, 9):
        print(
            "This launcher expects Python 3.9. Use a Python 3.9 interpreter or .venv39."
        )
        return 1
    if struct.calcsize("P") != 8:
        print("64-bit Python is required for the CPU wheel bundle.")
        return 1
    missing = []
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            version = getattr(m, "__version__", "0.0")
            print("{}: {}".format(name, version))
            numeric = tuple(int(x) for x in re.findall(r"\d+", version)[:2])
            if numeric < MINIMUM[name]:
                missing.append(name)
                print("  Version too old; use the Python 3.9 wheel bundle.")
        except Exception as exc:
            missing.append(name)
            print("{}: UNAVAILABLE ({})".format(name, exc))
    if missing:
        print("Missing/unusable packages: " + ", ".join(missing))
        print(
            "Run python\\setup_offline.bat D:\\path\\to\\wheelhouse once, then start.bat."
        )
        print("No package download was attempted.")
        return 1
    print("Dependencies available. No network access used.")
    return 0


if __name__ == "__main__":
    sys.exit(check())
