import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from pygsp import plotting


def latex_sci(val: float, decimals: int = 2) -> str:
    """Converts a value to LaTeX scientific notation A x 10^{B}."""
    if val == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(val))))
    mantissa = val / 10**exponent
    return rf"{mantissa:.{decimals}f} \times 10^{{{exponent}}}"


def latex_log_formatter(y: float, pos: int) -> str:
    """Custom formatter to render tick labels as LaTeX 10^{n}."""
    if y <= 0:
        return ""
    # Extract the exponent using log10
    n = int(np.round(np.log10(y)))
    return f"$10^{{{n}}}$"


def plot_setup() -> None:
    plotting.BACKEND = "matplotlib"
    plt.style.use(["science"])
    # TODO match font with document
