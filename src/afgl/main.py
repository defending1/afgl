import sys

import afgl.test_2 as t_2
from afgl.util.plot import plot_setup


def run() -> None:
    plot_setup()
    # t_1.run()
    t_2.run()


if __name__ == "__main__":
    # This block runs only if you call: python -m my_package.main
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
