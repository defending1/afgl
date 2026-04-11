import sys

import scienceplots  # noqa: F401

# Requires latex installed
import afgl.example_1 as ex1
import afgl.example_2 as ex2
from afgl.util.plot import plot_setup


def run() -> None:
    plot_setup()
    # example_1()
    ex1.run()
    ex2.run()


if __name__ == "__main__":
    # This block runs only if you call: python -m my_package.main
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
