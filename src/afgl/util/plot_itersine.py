import matplotlib.pyplot as plt
from pygsp import filters


def plot_itersine(G, Nf=7):
    G.estimate_lmax()
    g = filters.Itersine(G, Nf=Nf)

    g.plot()
    plt.savefig("./out/itersine_filterbank.pdf", bbox_inches="tight")
