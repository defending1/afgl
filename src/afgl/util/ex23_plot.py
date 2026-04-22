import matplotlib.pyplot as plt
import pandas as pd


def plot():
    df = pd.read_pickle("./out/ex_23.pkl")
    x = df["Duration (s) ex 2"]
    y = df["Duration (s) ex 3"]
    print(df)

    plt.figure(figsize=(8, 6))
    plt.loglog(x, y, "o", color="blue", label="Raw Data")
    plt.xlabel("Duration [s] ($N$ varies)")
    plt.ylabel("Duration [s] ($p$ varies)")
    plt.title("Performance correlation between $N$ and $p$.")

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig("./out/correlation_23.pdf", bbox_inches="tight")
