import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth


# ------------------------------------------------------------
# Data: real NASA GISTEMP v4 annual global land-ocean anomaly
# Source: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
# (annual mean column J-D, deviation from the 1951-1980 mean, deg C)
# ------------------------------------------------------------
def load_data():
    years = np.array([2013, 2014, 2015, 2016, 2017, 2018,
                      2019, 2020, 2021, 2022, 2023, 2024], dtype=float)
    temperature = np.array([0.68, 0.75, 0.90, 1.01, 0.92, 0.85,
                            0.98, 1.01, 0.85, 0.89, 1.17, 1.29], dtype=float)
    return years, temperature


# ------------------------------------------------------------
# Matrix representation:
#   X = data matrix, rows are (year, temperature)
#   B = design matrix [1, year],  t = temperature vector
#   The model B beta ~ t with beta = [c, m]^T means
#   temperature ~ m*year + c.
# ------------------------------------------------------------
def matrix_representation(years, temperature):
    X = np.column_stack((years, temperature))
    n = len(years)
    B = np.column_stack((np.ones(n), years))
    t = temperature.copy()
    return X, B, t


# ------------------------------------------------------------
# Least squares via orthogonal-projection method:
#   1. Q = orth(B)        -> orthonormal basis of col(B)
#   2. P = Q Q^T          -> projector onto col(B)   (P^2 = P, P^T = P)
#   3. t_hat = P t        -> closest vector to t in col(B)
#   t_hat = B beta, so lstsq(B, t) gives the slope/intercept.
# ------------------------------------------------------------
def least_squares_analysis(years, temperature, B, t):
    Q = orth(B)
    P = Q @ Q.T
    t_hat = P @ t

    # sanity checks shown on the results figure: Q^T Q ~ I, P^2 ~ P
    err_orth = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))
    err_idem = np.linalg.norm(P @ P - P)
    # Project 8 check: the orthonormal columns of Q span the same space as B
    rank_QB = (np.linalg.matrix_rank(Q),
               np.linalg.matrix_rank(np.column_stack((Q, B))))

    beta, *_ = np.linalg.lstsq(B, t, rcond=None)
    c, m = beta[0], beta[1]

    predicted = m * years + c
    residuals = temperature - predicted
    sse = np.sum(residuals ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_tot = np.sum((temperature - np.mean(temperature)) ** 2)
    r2 = 1 - sse / ss_tot

    return {"c": c, "m": m, "Q": Q, "P": P, "t_hat": t_hat,
            "predicted": predicted, "residuals": residuals,
            "sse": sse, "rmse": rmse, "r2": r2,
            "err_orth": err_orth, "err_idem": err_idem, "rank_QB": rank_QB}


# ------------------------------------------------------------
# PCA by hand:
#   1. center: Z = X - mu
#   2. covariance C = Z^T Z / (n-1)   (symmetric -> use eigh)
#   3. eigenpairs sorted by eigenvalue, descending
#   4. v1 = eigenvector of largest eigenvalue  -> principal direction
#   Projected points: mu + (z_i . v1) v1
# ------------------------------------------------------------
def pca_analysis(X):
    mu = np.mean(X, axis=0)
    Z = X - mu
    n = X.shape[0]
    C = (Z.T @ Z) / (n - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]       # descending
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]
    explained = eigenvalues[0] / np.sum(eigenvalues)

    projected = mu + np.array([(z @ v1) * v1 for z in Z])

    return {"mu": mu, "Z": Z, "C": C, "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors, "v1": v1, "v2": v2,
            "explained": explained, "projected": projected}


# PCA line: x(s) = mu + s*v1, s spanning the projections of the data
def pca_parametric_line(X, mu, v1):
    s = (X - mu) @ v1
    pad = 0.25 * (s.max() - s.min())
    s_line = np.linspace(s.min() - pad, s.max() + pad, 100)
    line = mu[:, None] + np.outer(v1, s_line)
    return line[0], line[1]


# Angle between LS direction (1, m) and PCA direction v1
def angle_between(ls, pca):
    d_ls = np.array([1.0, ls["m"]])
    d_ls /= np.linalg.norm(d_ls)
    d_pca = pca["v1"] / np.linalg.norm(pca["v1"])
    cos_theta = np.clip(abs(d_ls @ d_pca), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def plot_data(years, temperature):
    plt.figure()
    plt.get_current_fig_manager().set_window_title("1. Original Data")
    plt.scatter(years, temperature, color="black")
    plt.title("Original Temperature Data")
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (°C)")
    plt.grid(True)


def plot_results(ls, pca, theta_deg):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                   gridspec_kw={"height_ratios": [1.6, 1],
                                                "hspace": 0.15})
    fig.subplots_adjust(top=0.93, bottom=0.03)
    for ax in (ax1, ax2):
        ax.axis("off")
    fig.canvas.manager.set_window_title("2. Numerical Results")

    # ---- top: metric table ----
    rows = [
        ["Slope m",          f"{ls['m']:.5f}",                          "-"],
        ["Intercept c",      f"{ls['c']:.3f}",                          "-"],
        ["RMSE",             f"{ls['rmse']:.4f}",                       "-"],
        ["R²",               f"{ls['r2']:.4f}",                         "-"],
        ["||QᵀQ − I||",      f"{ls['err_orth']:.2e}",                   "-"],
        ["||P² − P||",       f"{ls['err_idem']:.2e}",                   "-"],
        ["Principal dir v1", "-",                       f"[{pca['v1'][0]:.4f}, {pca['v1'][1]:.4f}]"],
        ["||v1||",           "-",                       f"{np.linalg.norm(pca['v1']):.4f}"],
        ["Eigenvalues",      "-",           f"{pca['eigenvalues'][0]:.4f}, {pca['eigenvalues'][1]:.4f}"],
        ["v1 · v2",          "-",                       f"{pca['v1'] @ pca['v2']:.2e}"],
        ["Explained var.",   "-",                       f"{pca['explained']*100:.2f}%"],
        ["Angle LS↔PCA",     f"{theta_deg:.4f}°",                       f"{theta_deg:.4f}°"],
    ]
    tbl = ax1.table(cellText=rows,
                    colLabels=["Quantity", "Least Squares", "PCA"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#d9e6f2")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f2f2")
    fig.suptitle("Numerical Results", y=0.98, fontsize=13, weight="bold")

    text = (
        "Least squares:  T = m*Year + c,  minimizes  ||B*beta - t||^2 (VERTICAL errors)\n"
        "PCA:            x(s) = mu + s*v1,            maximizes variance along v1 (PERPENDICULAR errors)\n"
        f"Angle between directions: {theta_deg:.4f} deg (~0: year-spread dominates,\n"
        "so PCA ~ LS on this data - the error GEOMETRY is what differs)\n"
        "To predict T from Year: use least squares.   For geometry: use PCA."
    )
    ax2.text(0.02, 0.90, text, va="top", family="monospace",
             fontsize=10, linespacing=1.9)


def plot_least_squares(years, temperature, ls):
    plt.figure()
    plt.get_current_fig_manager().set_window_title("3. Least-Squares Trend Line")
    plt.scatter(years, temperature, color="black", label="Data")
    plt.plot(years, ls["predicted"], color="tab:blue",
             label=f"Least-squares: T = {ls['m']:.4f}·Year + ({ls['c']:.2f})")
    for y, a, p in zip(years, temperature, ls["predicted"]):
        plt.plot([y, y], [a, p], color="gray", linewidth=0.8)
    plt.title("Least-Squares Trend Line (vertical errors)")
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (°C)")
    plt.legend()
    plt.grid(True)


def plot_pca(years, temperature, pca, line_x, line_y):
    plt.figure()
    plt.get_current_fig_manager().set_window_title("4. PCA Principal Direction")
    plt.scatter(years, temperature, color="black", label="Data")
    plt.plot(line_x, line_y, color="tab:red", label="PCA: x(s) = μ + s·v1")
    plt.scatter([pca["mu"][0]], [pca["mu"][1]], color="tab:red",
                marker="x", s=100, label="Centroid μ")
    plt.title("PCA Principal Direction")
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (°C)")
    plt.legend()
    plt.grid(True)


def plot_pca_projections(years, temperature, X, pca, line_x, line_y):
    plt.figure()
    plt.get_current_fig_manager().set_window_title("5. PCA Projections")
    plt.scatter(years, temperature, color="black", label="Data")
    plt.scatter(pca["projected"][:, 0], pca["projected"][:, 1],
                color="tab:purple", label="PCA projections")
    for (x0, y0), (x1, y1) in zip(X, pca["projected"]):
        plt.plot([x0, x1], [y0, y1], color="gray", linewidth=0.8)
    plt.plot(line_x, line_y, color="tab:red", label="PCA line")
    plt.scatter([pca["mu"][0]], [pca["mu"][1]], color="tab:red",
                marker="x", s=100, label="Centroid μ")
    plt.title("Orthogonal Projection onto the Principal Direction")
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (°C)")
    plt.legend()
    plt.grid(True)


def plot_comparison(years, temperature, ls, pca, line_x, line_y):
    plt.figure()
    plt.get_current_fig_manager().set_window_title("6. Comparison: LS vs PCA")
    plt.scatter(years, temperature, color="black", label="Data")
    plt.plot(years, ls["predicted"], color="tab:blue", label="Least-squares line")
    plt.plot(line_x, line_y, color="tab:red", label="PCA line")
    plt.scatter([pca["mu"][0]], [pca["mu"][1]], color="tab:red",
                marker="x", s=100, label="Centroid μ")
    plt.title("Least-Squares Trend Line vs PCA Principal Direction")
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (°C)")
    plt.legend()
    plt.grid(True)


def main():
    years, temperature = load_data()
    X, B, t = matrix_representation(years, temperature)
    ls = least_squares_analysis(years, temperature, B, t)
    pca = pca_analysis(X)
    theta_deg = angle_between(ls, pca)
    line_x, line_y = pca_parametric_line(X, pca["mu"], pca["v1"])

    plot_data(years, temperature)
    plot_results(ls, pca, theta_deg)
    plot_least_squares(years, temperature, ls)
    plot_pca(years, temperature, pca, line_x, line_y)
    plot_pca_projections(years, temperature, X, pca, line_x, line_y)
    plot_comparison(years, temperature, ls, pca, line_x, line_y)

    plt.show()


if __name__ == "__main__":
    main()
