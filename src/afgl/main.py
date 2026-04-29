import sys

import afgl.ex1 as ex1
from afgl.ex23 import Ex23
from afgl.plot.ex23 import plot_ex23
from afgl.plot.setup import plot_setup


def run() -> None:
    plot_setup()
    ex1.run()
    Ex23(3, 1).run()
    plot_ex23()


if __name__ == "__main__":
    # This block runs only if you call: python -m my_package.main
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
