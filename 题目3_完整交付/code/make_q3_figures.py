"""Generate the Question-3 figures from the saved numerical results.

This script does not run an optimizer.  It reads results/q3_result.json and
the two prediction CSV files, then recreates only deterministic diagnostic
curves needed by the figures.  Every output is written as both PDF and PNG.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from solve_q3 import (
    SIC_CONSTANTS,
    Spectrum,
    load_spectrum,
    round_trip_strength,
    sic_index,
    si_drude_index,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf.relative_to(ROOT)), "png": str(png.relative_to(ROOT))}


def figure_input_and_band(result: dict) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    band = result["si_band_selection"]["selected"]["band_cm_inv"]
    colors = {3: "#1f77b4", 4: "#d62728"}
    for attachment in (3, 4):
        spectrum = load_spectrum(attachment)
        ax = axes[0]
        ax.plot(
            spectrum.sigma_cm_inv,
            spectrum.reflectance_percent,
            color=colors[attachment],
            linewidth=0.55,
            alpha=0.75,
            label=f"Attachment {attachment} ({int(spectrum.angle_deg)} deg)",
        )
    axes[0].axvspan(band[0], band[1], color="#2ca02c", alpha=0.14, label="selected fit band")
    axes[0].set_title("Silicon input spectra and selected common band")
    axes[0].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0].set_ylabel("Reflectance (%)")
    axes[0].legend(loc="upper left")

    candidates = result["si_band_selection"]["candidates"]
    labels = [f"{int(row['band_cm_inv'][0])}--{int(row['band_cm_inv'][1])}" for row in candidates]
    scores = [row["score"] for row in candidates]
    x = np.arange(len(labels))
    bar_colors = ["#f0ad4e" if label == f"{int(band[0])}--{int(band[1])}" else "#9ecae1" for label in labels]
    axes[1].bar(x, scores, color=bar_colors, edgecolor="#244a64")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("dual-angle stability score")
    axes[1].set_title("Band selection score")
    axes[1].text(
        0.98,
        0.96,
        f"selected: {int(band[0])}--{int(band[1])} $\\mathrm{{cm}}^{{-1}}$",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    return save_figure(fig, "fig_q3_input_and_band")


def figure_si_model_comparison(result: dict) -> dict[str, str]:
    rows = read_csv(RESULTS_DIR / "q3_si_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=False, constrained_layout=True)
    styles = {
        "two_beam_percent": ("#7f7f7f", "two-beam", "--"),
        "full_multibeam_percent": ("#ff7f0e", "full Airy eta=1", "-"),
        "partial_multibeam_percent": ("#2ca02c", "partial multibeam", "-"),
    }
    for col, attachment in enumerate((3, 4)):
        selected_rows = [row for row in rows if int(row["attachment"]) == attachment]
        sigma = np.asarray([row["wavenumber_cm_inv"] for row in selected_rows])
        observed = np.asarray([row["observed_percent"] for row in selected_rows])
        ax_fit, ax_res = axes[0, col], axes[1, col]
        ax_fit.plot(sigma, observed, color="#bdbdbd", linewidth=0.7, label="measured")
        for key, (color, label, linestyle) in styles.items():
            ax_fit.plot(sigma, [row[key] for row in selected_rows], color=color, linewidth=1.0, linestyle=linestyle, label=label)
        ax_fit.set_title(f"Attachment {attachment} ({10 if attachment == 3 else 15} deg)")
        ax_fit.set_ylabel("Reflectance (%)")
        ax_fit.legend(loc="best")
        residual = np.asarray([row["selected_residual_percent"] for row in selected_rows])
        ax_res.plot(sigma, residual, color="#1f4e79", linewidth=0.75)
        ax_res.axhline(0.0, color="#d62728", linewidth=0.8)
        ax_res.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax_res.set_ylabel("selected residual (%)")
    fig.suptitle("Silicon two-beam / full-Airy / partial-multibeam comparison", y=1.02)
    return save_figure(fig, "fig_q3_si_model_comparison")


def figure_model_metrics(result: dict) -> dict[str, str]:
    modes = ("two_beam", "full_multibeam", "partial_multibeam")
    labels = ("two-beam", "full Airy", "partial")
    mse = [result["si_models"][mode]["optimizer"]["full_mse_percent_squared"] for mode in modes]
    bic = [result["si_models"][mode]["information_criteria"]["bic"] for mode in modes]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), constrained_layout=True)
    x = np.arange(3)
    axes[0].bar(x, mse, color=["#9e9e9e", "#ff9f40", "#2ca02c"], edgecolor="#333333")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("full-band MSE (% point)$^2$")
    axes[0].set_title("Fit error")
    for i, value in enumerate(mse):
        axes[0].text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(x, bic, color=["#9e9e9e", "#ff9f40", "#2ca02c"], edgecolor="#333333")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("BIC (lower is better)")
    axes[1].set_title("Information criterion")
    for i, value in enumerate(bic):
        axes[1].text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    return save_figure(fig, "fig_q3_model_metrics")


def figure_round_trip(result: dict) -> dict[str, str]:
    si_result = result["si_models"][result["si_model_selection"]["selected_model"]]
    si_params = si_result["parameters"]
    sic_params = result["sic_correction"]["parameters"]
    si_band = tuple(result["si_band_selection"]["selected"]["band_cm_inv"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)

    for ax, material, attachments in zip(axes, ("Si", "SiC"), ((3, 4), (1, 2))):
        for attachment in attachments:
            spectrum = load_spectrum(attachment)
            if material == "Si":
                valid = (spectrum.sigma_cm_inv >= si_band[0]) & (spectrum.sigma_cm_inv <= si_band[1])
                sigma_plot = spectrum.sigma_cm_inv[valid]
                n_layer = si_drude_index(sigma_plot, si_params["omega_p_layer_cm_inv"], 0.0)
                n_substrate = si_drude_index(sigma_plot, si_params["omega_p_substrate_cm_inv"], 0.0)
                d_um = si_params["d_um"]
            else:
                sigma_plot = spectrum.sigma_cm_inv
                n_layer = sic_index(
                    sigma_plot,
                    sic_params["gamma_L_cm_inv"],
                    sic_params["gamma_T_cm_inv"],
                    sic_params["omega_p_layer_cm_inv"],
                    sic_params["gamma_p_layer_cm_inv"],
                )
                n_substrate = sic_index(
                    sigma_plot,
                    sic_params["gamma_L_cm_inv"],
                    sic_params["gamma_T_cm_inv"],
                    sic_params["omega_p_substrate_cm_inv"],
                    sic_params["gamma_p_substrate_cm_inv"],
                )
                d_um = sic_params["d_um"]
            rho = round_trip_strength(
                sigma_plot, n_layer, n_substrate, d_um, spectrum.angle_deg
            )
            ax.plot(sigma_plot, rho, linewidth=0.9, label=f"Attachment {attachment}")
        ax.set_title(f"{material}: successive-return amplitude ratio")
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel(r"$\rho=|r_{10}r_{12}e^{i\Delta\phi}|$")
        ax.legend(loc="best")
    fig.suptitle("Necessary amplitude condition for observable higher-order returns", y=1.02)
    return save_figure(fig, "fig_q3_round_trip_strength")


def figure_substrate_sensitivity(result: dict) -> dict[str, str]:
    sensitivity = result["si_substrate_sensitivity"]
    records = sensitivity["records"]
    x = np.asarray([row["fixed_omega_p_substrate_cm_inv"] for row in records])
    d = np.asarray([row["d_um"] for row in records])
    rel_mse = np.asarray([row["relative_mse_to_selected"] for row in records])
    reference_d = sensitivity["reference_d_um"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), constrained_layout=True)
    axes[0].plot(x, d, "o-", color="#1f77b4")
    axes[0].axhline(reference_d, color="#d62728", linestyle="--", linewidth=0.9, label="selected fit")
    axes[0].set_xlabel(r"fixed $\omega_{p,substrate}$ (cm$^{-1}$)")
    axes[0].set_ylabel("conditional thickness (um)")
    axes[0].set_title("Thickness sensitivity")
    axes[0].legend(loc="best")
    axes[1].plot(x, rel_mse, "o-", color="#2ca02c")
    axes[1].axhline(1.20, color="#d62728", linestyle="--", linewidth=0.9, label="comparable threshold")
    axes[1].fill_between(x, 0, 1.20, color="#2ca02c", alpha=0.08)
    axes[1].set_xlabel(r"fixed $\omega_{p,substrate}$ (cm$^{-1}$)")
    axes[1].set_ylabel("MSE / selected MSE")
    axes[1].set_title("Conditional fit quality")
    axes[1].legend(loc="best")
    return save_figure(fig, "fig_q3_substrate_sensitivity")


def figure_sic_correction(result: dict) -> dict[str, str]:
    rows = read_csv(RESULTS_DIR / "q3_sic_correction_predictions.csv")
    reference = np.asarray(result["sic_correction"]["reference_q2_parameters"]["d_um"])
    corrected = result["sic_correction"]["parameters"]["d_um"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    colors = {1: "#1f77b4", 2: "#d62728"}
    for attachment in (1, 2):
        selected = [row for row in rows if int(row["attachment"]) == attachment]
        sigma = np.asarray([row["wavenumber_cm_inv"] for row in selected])
        observed = np.asarray([row["observed_percent"] for row in selected])
        prediction = np.asarray([row["exact_multibeam_corrected_percent"] for row in selected])
        axes[0].plot(sigma, observed, color=colors[attachment], linewidth=0.5, alpha=0.65, label=f"measured {attachment}")
        axes[0].plot(sigma, prediction, color=colors[attachment], linewidth=0.9, label=f"corrected fit {attachment}")
        axes[1].plot(sigma, prediction - observed, color=colors[attachment], linewidth=0.75, label=f"attachment {attachment}")
    axes[0].set_title("SiC exact-Airy correction fit")
    axes[0].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0].set_ylabel("Reflectance (%)")
    axes[0].legend(loc="best")
    axes[1].axhline(0.0, color="#333333", linewidth=0.7)
    axes[1].set_title("Corrected residuals")
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("residual (%)")
    axes[1].legend(loc="best")
    fig.suptitle(f"SiC correction: d {reference:.6f} -> {corrected:.6f} um", y=1.02)
    return save_figure(fig, "fig_q3_sic_correction")


def figure_workflow() -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(13.0, 3.3))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    boxes = [
        (0.3, 1.25, 1.8, 0.85, "Input\nAttachments 1--4"),
        (2.45, 1.25, 1.8, 0.85, "Audit\nremove endpoint only"),
        (4.6, 1.25, 1.8, 0.85, "Si band\nFFT + peak check"),
        (6.75, 1.25, 1.8, 0.85, "Fresnel\nAiry series"),
        (8.9, 1.25, 1.8, 0.85, "Model compare\nBIC + eta"),
        (11.05, 1.25, 1.6, 0.85, "Report\nfit + correction"),
    ]
    colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#f3e8ff", "#fee2e2", "#dbeafe"]
    for index, (x, y, w, h, text_value) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.0,
            edgecolor="#244a64",
            facecolor=colors[index],
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text_value, ha="center", va="center", fontsize=9)
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            ax.annotate("", xy=(next_x - 0.1, y + h / 2), xytext=(x + w + 0.1, y + h / 2), arrowprops={"arrowstyle": "->", "color": "#244a64", "lw": 1.1})
    ax.text(7.65, 2.65, "eta = 0: two-beam   |   eta = 1: exact coherent sum   |   0 < eta < 1: effective partial coherence", ha="center", va="center", fontsize=9, color="#444444")
    ax.text(6.5, 0.55, "Q3 independent reproducible workflow", ha="center", va="center", fontsize=12, weight="bold")
    return save_figure(fig, "fig_q3_workflow")


def main() -> None:
    result = json.loads((RESULTS_DIR / "q3_result.json").read_text(encoding="utf-8"))
    manifest = {
        "figures": {
            "input_and_band": figure_input_and_band(result),
            "si_model_comparison": figure_si_model_comparison(result),
            "model_metrics": figure_model_metrics(result),
            "round_trip_strength": figure_round_trip(result),
            "substrate_sensitivity": figure_substrate_sensitivity(result),
            "sic_correction": figure_sic_correction(result),
            "workflow": figure_workflow(),
        },
        "source": [
            "results/q3_result.json",
            "results/q3_si_predictions.csv",
            "results/q3_sic_correction_predictions.csv",
        ],
        "note": "Figures are deterministic renderings of the saved Q3 results; this script does not optimize parameters.",
    }
    (FIGURES_DIR / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
