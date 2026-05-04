import sys

import afgl.ex1 as ex1
from afgl.ex23 import Ex23
from afgl.lanczos_vs_arnoldi import LanczosVsArnoldi
from afgl.plot.ex23 import plot_ex23
from afgl.plot.setup import plot_setup


def run() -> None:
    plot_setup()
    LanczosVsArnoldi().run()

    ex1.run()
    Ex23_ = Ex23(3)
    Ex23_.run()
    Ex23_.print_group_table()
    plot_ex23()


if __name__ == "__main__":
    # This block runs only if you call: python -m my_package.main
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
