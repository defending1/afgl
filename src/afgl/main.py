import sys

from afgl.ex_23 import Ex23
from afgl.util.plot import plot_setup


def run() -> None:
    plot_setup()
    # ex_1.run()
    Ex23(3, 1).run()
    # plot()


if __name__ == "__main__":
    # This block runs only if you call: python -m my_package.main
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
