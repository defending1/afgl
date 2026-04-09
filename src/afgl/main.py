# src/afgl/main.py
import sys

import matplotlib.pyplot as plt
import numpy as np
from pygsp import graphs, plotting


def test_plot():
    plotting.BACKEND = "matplotlib"
    plt.style.use(["science"])
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    # G = graphs.ErdosRenyi()
    G = graphs.Sensor()

    G.set_coordinates()

    signal = np.sin(G.coords[:, 0] * 10)

    G.plot(signal, ax=ax, vertex_size=15, edge_width=0.5, edge_color="gray")
    ax.set_title(r"Sensor Network $\mathcal{G} = (\mathcal{V}, \mathcal{E})$")
    ax.set_axis_off()
    plt.savefig("./out/test.pdf", bbox_inches="tight")


def run():
    test_plot()


if __name__ == "__main__":
    # This block runs only if you call: python -m my_package.main
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
