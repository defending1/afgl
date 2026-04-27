import matplotlib.pyplot as plt
import pandas as pd


def plot():
    df = pd.read_pickle("./out/ex_23.pkl")
    x = df["Time 2"]
    y = df["Time 3"]
    print(x.var())

    plt.figure(figsize=(8, 6))
    plt.loglog(x, y, "o", color="blue", label="Raw Data")
    plt.xlabel("Time [s] ($N$ varies)")
    plt.ylabel("Time [s] ($p$ varies)")
    plt.title("Performance correlation between $N$ and $p$.")

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig("./out/correlation_23.pdf", bbox_inches="tight")
