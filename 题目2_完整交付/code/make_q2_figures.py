"""Generate the figures used by the Question-2 LaTeX solution.

Run this file after ``solve_q2.py``.  It only reads ``results`` and the two
workbooks; it does not rerun the optimiser.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.signal import savgol_filter

from solve_q2 import DEFAULT_DATA_DIR, RESULT_DIR, ROOT, load_spectrum, period_initialisation


FIGURE_DIR = ROOT / "figures"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
            "savefig.dpi": 240,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def figure_initial_peaks() -> None:
    spectra = [load_spectrum(1), load_spectrum(2)]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.8), sharex=True)
    for ax, spectrum in zip(axes, spectra):
        initial = period_initialisation(spectrum)
        mask = (spectrum.sigma_cm_inv >= 1300.0) & (spectrum.sigma_cm_inv <= 3500.0)
        x = spectrum.sigma_cm_inv[mask]
        y = spectrum.reflectance_percent[mask]
        smooth = savgol_filter(y, initial["smoothing"]["window_points"], 3)
        ax.scatter(x, y, s=2.0, color="#b9c1cc", alpha=0.58, label="measured")
        ax.plot(x, smooth, color="#1d4e89", linewidth=1.2, label="SG-smoothed")
        peaks = np.asarray(initial["peaks_cm_inv"], dtype=float)
        peak_values = np.interp(peaks, x, smooth)
        ax.scatter(peaks, peak_values, s=34, color="#c0392b", zorder=4, label="detected peaks")
        ax.set_ylabel("Reflectance (%)")
        ax.set_title(f"Attachment {spectrum.attachment}  ({spectrum.angle_deg:.0f} deg)")
        ax.legend(loc="upper left", ncol=3, frameon=True)
        ax.text(
            0.99,
            0.06,
            f"median spacing = {initial['median_spacing_cm_inv']:.3f} cm$^{{-1}}$\n"
            f"period initial d = {initial['d_initial_um']:.4f} um",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.86},
        )
    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[-1].set_xlim(1300, 3500)
    fig.suptitle("Question-1 period initialisation for the two SiC spectra", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig_q2_initial_peaks")


def read_predictions() -> dict[int, dict[str, np.ndarray]]:
    path = RESULT_DIR / "q2_predictions.csv"
    grouped: dict[int, dict[str, list[float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            attachment = int(row["attachment"])
            grouped.setdefault(
                attachment,
                {
                    "angle": [],
                    "sigma": [],
                    "observed": [],
                    "physical": [],
                    "fitted": [],
                    "residual": [],
                },
            )
            grouped[attachment]["angle"].append(float(row["angle_deg"]))
            grouped[attachment]["sigma"].append(float(row["wavenumber_cm_inv"]))
            grouped[attachment]["observed"].append(float(row["observed_percent"]))
            grouped[attachment]["physical"].append(float(row["physical_model_percent"]))
            grouped[attachment]["fitted"].append(float(row["fitted_percent"]))
            grouped[attachment]["residual"].append(float(row["residual_percent"]))
    return {
        key: {name: np.asarray(values, dtype=float) for name, values in value.items()}
        for key, value in grouped.items()
    }


def figure_fit_residual() -> None:
    grouped = read_predictions()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharex="col")
    for column, attachment in enumerate((1, 2)):
        row = grouped[attachment]
        step = max(1, row["sigma"].size // 3000)
        x = row["sigma"][::step]
        observed = row["observed"][::step]
        fitted = row["fitted"][::step]
        residual = row["residual"][::step]
        axes[0, column].plot(x, observed, color="#aeb7c4", linewidth=0.62, label="measured")
        axes[0, column].plot(x, fitted, color="#d35400", linewidth=1.0, label="joint fit")
        axes[0, column].set_title(f"Attachment {attachment} ({row['angle'][0]:.0f} deg)")
        axes[0, column].set_ylabel("Reflectance (%)")
        axes[0, column].legend(loc="upper right")
        axes[1, column].plot(x, residual, color="#2c3e50", linewidth=0.65)
        axes[1, column].axhline(0.0, color="#c0392b", linewidth=0.8)
        axes[1, column].set_xlabel("Wavenumber (cm$^{-1}$)")
        axes[1, column].set_ylabel("Residual (%)")
    fig.suptitle("SiC two-beam joint fit and residuals", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig_q2_fit_residual")


def figure_thickness_profile() -> None:
    path = RESULT_DIR / "q2_thickness_profile.csv"
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
    fit = json.loads((RESULT_DIR / "q2_fit.json").read_text(encoding="utf-8"))
    best_d = float(fit["joint_fit"]["parameters"]["d_um"])
    interval = fit["thickness_profile"]["summary"]["interval_um"]
    d = np.asarray(data["d_um"], dtype=float)
    mse = np.asarray(data["mse_percent_squared"], dtype=float)
    rel = np.asarray(data["relative_to_best_mse"], dtype=float) * 100.0
    order = np.argsort(d)
    d, mse, rel = d[order], mse[order], rel[order]

    fig, ax1 = plt.subplots(figsize=(8.6, 5.0))
    ax1.plot(d, mse, "o-", color="#1d4e89", linewidth=1.4, markersize=3.5, label="profile MSE")
    ax1.set_xlabel("Fixed thickness d (um)")
    ax1.set_ylabel("Re-optimised MSE (%$^2$)", color="#1d4e89")
    ax1.tick_params(axis="y", labelcolor="#1d4e89")
    ax1.axvline(best_d, color="#c0392b", linestyle="--", linewidth=1.1, label=f"best d = {best_d:.5f} um")
    ax1.axvspan(interval[0], interval[1], color="#f1c40f", alpha=0.20, label="1% MSE sensitivity interval")
    ax2 = ax1.twinx()
    ax2.plot(d, rel, color="#27ae60", linewidth=1.0, alpha=0.85, label="relative MSE increase")
    ax2.axhline(1.0, color="#27ae60", linestyle=":", linewidth=1.0, label="1% threshold")
    ax2.set_ylabel("Relative MSE increase (%)", color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title("Thickness profile with nuisance parameters re-optimised")
    fig.tight_layout()
    save(fig, "fig_q2_thickness_profile")


def figure_sensitivity() -> None:
    payload = json.loads((RESULT_DIR / "q2_fit.json").read_text(encoding="utf-8"))
    records = payload["material_constant_sensitivity"]["records"]
    baseline = float(payload["joint_fit"]["parameters"]["d_um"])
    names = ["eps_inf", "omega_L_cm_inv", "omega_T_cm_inv"]
    labels = [r"$\epsilon_\infty$", r"$\omega_L$", r"$\omega_T$"]
    positions = np.arange(len(names), dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for position, name in zip(positions, names):
        values = [
            float(row["conditional_d_um"])
            for row in records
            if row["perturbed_parameter"] == name
        ]
        changes = [
            100.0 * float(row["relative_change"])
            for row in records
            if row["perturbed_parameter"] == name
        ]
        ax.plot([position - 0.12, position + 0.12], values, "o-", linewidth=1.4, markersize=5)
        ax.text(position - 0.12, values[0], f"{changes[0]:+.0f}%", ha="right", va="bottom", fontsize=8)
        ax.text(position + 0.12, values[1], f"{changes[1]:+.0f}%", ha="left", va="bottom", fontsize=8)
    ax.axhline(baseline, color="#c0392b", linestyle="--", linewidth=1.1, label=f"baseline d = {baseline:.5f} um")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Conditional thickness (um)")
    ax.set_title("Thickness sensitivity to fixed SiC material constants")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "fig_q2_material_sensitivity")


def figure_workflow() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    nodes = [
        (0.2, 1.55, 1.55, 0.95, "Input\nattachments 1/2", "#e8f1fb"),
        (2.05, 1.55, 1.55, 0.95, "Audit\nremove endpoint only", "#eaf7ef"),
        (3.90, 1.55, 1.55, 0.95, "Peak spacing\ninitial d", "#fff4dc"),
        (5.75, 1.55, 1.75, 0.95, "Two-beam\nFresnel + LD", "#f4eafb"),
        (7.80, 1.55, 1.55, 0.95, "DE\nsubsample", "#fcebea"),
        (9.65, 1.55, 1.15, 0.95, "L-BFGS-B\nfull", "#e8f1fb"),
    ]
    for x, y, w, h, label, color in nodes:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            linewidth=1.0,
            edgecolor="#34495e",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    for left, right in zip(nodes[:-1], nodes[1:]):
        x0 = left[0] + left[2]
        x1 = right[0]
        ax.annotate("", xy=(x1 - 0.07, left[1] + left[3] / 2), xytext=(x0 + 0.07, left[1] + left[3] / 2), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#34495e"})
    ax.annotate(
        "",
        xy=(6.65, 1.45),
        xytext=(6.65, 0.75),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#7f8c8d"},
    )
    ax.text(6.65, 0.46, "two angles share d;\nscale/offset solved analytically", ha="center", va="center", fontsize=9, color="#566573")
    ax.annotate(
        "",
        xy=(10.23, 2.55),
        xytext=(10.23, 3.18),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#7f8c8d"},
    )
    ax.text(10.23, 3.53, "metrics, profile,\nsensitivity, output", ha="center", va="center", fontsize=9, color="#566573")
    ax.set_title("Question-2 reproducible computation workflow", pad=8)
    fig.tight_layout()
    save(fig, "fig_q2_workflow")


def main() -> None:
    style()
    figure_initial_peaks()
    figure_fit_residual()
    figure_thickness_profile()
    figure_sensitivity()
    figure_workflow()
    manifest = {
        "figures": [
            {"stem": "fig_q2_initial_peaks", "purpose": "SG-smoothed spectra and period peaks"},
            {"stem": "fig_q2_fit_residual", "purpose": "two-angle fit and residuals"},
            {"stem": "fig_q2_thickness_profile", "purpose": "re-optimised thickness sensitivity profile"},
            {"stem": "fig_q2_material_sensitivity", "purpose": "material-constant perturbation check"},
            {"stem": "fig_q2_workflow", "purpose": "algorithm workflow"},
        ],
        "source": "code/make_q2_figures.py",
    }
    (FIGURE_DIR / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
