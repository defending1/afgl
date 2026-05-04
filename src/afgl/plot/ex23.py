import matplotlib.pyplot as plt
import pandas as pd


def plot_ex23():
    df2 = pd.read_pickle("./out/df2.pkl")
    df3 = pd.read_pickle("./out/df3.pkl")
    x = df2["Time"]
    y = df3["Time"]
    color_labels = df3["color"]
    n_values = df2["$N$"]
    p_values = df3["$p$"]

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    unique_labels = pd.Index(color_labels).unique()
    marker_map = {
        label: markers[idx % len(markers)] for idx, label in enumerate(unique_labels)
    }

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    with plt.style.context(["science", "grid"]):
        # Initialize figure and axes. SciencePlots handles default sizing,
        # but figsize can still be explicitly declared if needed.
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot data with label-based colors and markers.
        for label in unique_labels:
            mask = color_labels == label
            n_value = n_values.loc[mask].iloc[0]
            p_value = p_values.loc[mask].iloc[0]
            ax.loglog(
                x[mask],
                y[mask],
                marker=marker_map[label],
                linestyle="",
                alpha=0.8,
                color=label,
                label=f"$({n_value},{p_value})$",
            )

        ax.legend(
            title="$(N,p)$",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            frameon=False,
        )

        # Set labels and title
        ax.set_xlabel("Time [s] ($N$ varies)")
        ax.set_ylabel("Time [s] ($p$ varies)")
        ax.set_title("Performance Correlation Between $N$ and $p$")

        # Save and close
        plt.savefig("./out/correlation_23.pdf", bbox_inches="tight")
        plt.close()
