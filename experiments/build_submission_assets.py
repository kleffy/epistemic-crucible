"""Deterministically generate submission Tables 1--4 and Figures 1--4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ARM_ORDER = ("neutral", "cue_a", "cue_b", "mechanism_a", "mechanism_b")
ARM_LABELS = {
    "neutral": "Neutral",
    "cue_a": "Cue A",
    "cue_b": "Cue B",
    "mechanism_a": "Mechanism A",
    "mechanism_b": "Mechanism B",
}
MODEL_ORDER = ("qwen", "mistral")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def _write(path: pathlib.Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    if abs(value) < 0.5 * 10 ** (-digits):
        return "0"
    if abs(value - 1.0) < 0.5 * 10 ** (-digits):
        return "1"
    rendered = f"{value:.{digits}f}"
    if rendered.startswith("0."):
        return rendered[1:]
    if rendered.startswith("-0."):
        return "-" + rendered[2:]
    return rendered


def _signed(value: float) -> str:
    rendered = _fmt(value)
    return rendered if value <= 0 else f"+{rendered}"


def _ci(metric: dict[str, Any], *, denominator: str | None = None) -> str:
    den = denominator or f"$n={metric['denominator']}$"
    return (
        f"\\cellci{{{_fmt(metric['value'])}}}"
        f"{{{_fmt(metric.get('ci_low'))}}}{{{_fmt(metric.get('ci_high'))}}}"
        f"{{{den}}}"
    )


def _table_1(integrity: dict[str, Any]) -> str:
    counts = integrity["counts"]
    if not integrity["passed"] or integrity["violations"]:
        raise ValueError("integrity report did not pass")
    return rf"""
\begin{{table}}[t]
  \centering
  \small
  \begin{{tabular}}{{lrl}}
    \toprule
    Validation & Scale & Result \\
    \midrule
    Quartet invariants & {counts["invariant_seeds"]:,} base seeds & 0 violations \\
    Legal oracle execution & {counts["oracle_checks"]:,} cell--policy runs &
      100\% success \\
    Mechanism-axis public equality & all checked pairs & 100\% \\
    Cue-axis equality after color mask & all checked pairs & 100\% \\
    Scripted policy calibration & all reference policies & exact coordinates \\
    \bottomrule
  \end{{tabular}}
  \caption{{Pre-scientific integrity gates. Scientific outcomes cannot waive a
  failed row.}}
  \label{{tab:integrity}}
\end{{table}}
"""


def _table_2(bc: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {"cue": [], "mechanism": []}
    for run in bc:
        grouped[run["regime"]].append(run["metrics"])
    for regime, reports in grouped.items():
        if len(reports) != 5:
            raise ValueError(f"expected five {regime} BC runs")
        projection = {
            (
                report["aligned_success"]["value"],
                report["crossed_success"]["value"],
                report["mechanism_responsiveness"]["value"],
                report["cue_susceptibility"]["value"],
                report["mechanism_tracking"]["value"],
                report["cue_following"]["value"],
                report["detector_query_rate"]["value"],
            )
            for report in reports
        }
        if len(projection) != 1:
            raise ValueError(f"{regime} BC runs do not reproduce one coordinate")
    return r"""
\begin{table}[t]
  \centering
  \small
  \setlength{\tabcolsep}{4.5pt}
  \begin{tabular}{lrrrrrr}
    \toprule
    Policy & Align. & Crossed & $R_M$ & $S_C^{\rm commit}$ & Tracking & Query rate \\
    \midrule
    Cue-BC & 1.00 & 0.00 & 0.00 & 1.00 & $T_C=1.00$ & 0.00 \\
    Mechanism-BC & 1.00 & 1.00 & 1.00 & 0.00 & $T_M=1.00$ & 1.00 \\
    \bottomrule
  \end{tabular}
  \caption{Held-out construct validation on base seeds 100--163. Every value was
  reproduced across all five initialization seeds in each regime.}
  \label{tab:bc}
\end{table}
"""


def _table_3(inputs: dict[str, Any]) -> str:
    rows: list[str] = []
    for model_index, key in enumerate(MODEL_ORDER):
        model = inputs["models"][key]
        short_name = "Qwen" if key == "qwen" else "Mistral"
        for arm_index, arm in enumerate(ARM_ORDER):
            report = model["reports"][arm]
            committed = round(report["coverage"]["value"] * 384)
            model_cell = short_name if arm_index == 0 else ""
            diag = report["diagnostics"]
            row = " & ".join(
                (
                    model_cell,
                    ARM_LABELS[arm],
                    _ci(report["coverage"], denominator=f"{committed}/384"),
                    _ci(report["mechanism_accuracy"]),
                    _ci(report["cue_following"]),
                    _ci(report["cue_susceptibility"]),
                    _ci(report["detector_query_rate"]),
                    _ci(report["cell_success"]),
                    _ci(report["all_cells_success"]),
                    f"{diag['invalid_responses']}/{diag['length_finishes']}",
                )
            )
            rows.append("    " + row + r" \\")
        if model_index == 0:
            rows.append(r"    \midrule")
    return (
        r"""
\begin{table*}[t]
  \centering
  \begingroup
  \newcommand{\cellci}[4]{\shortstack{#1\\{\scriptsize[#2,#3]}\\{\scriptsize #4}}}
  \scriptsize
  \setlength{\tabcolsep}{2.0pt}
  \begin{tabular}{@{}llcccccccc@{}}
    \toprule
    Model & Arm & Coverage & $\mathrm{Acc}_M$ & $F_C$ & $S_C$ & Query &
      Cell success & All-six & Invalid / length \\
    \midrule
"""
        + "\n".join(rows)
        + r"""
    \bottomrule
  \end{tabular}
  \endgroup
  \caption{Frozen confirmatory outcomes. Each metric cell gives estimate, paired
  seed-clustered bootstrap 95\% interval, and denominator; coverage gives committed
  cells/384. $\mathrm{Acc}_M$ and $F_C$ condition on commitment, $S_C$ on paired
  commitments, query on all cells, and cell/all-six success on 64 base seeds.
  Diagnostics give invalid responses / length finishes out of 384 cells.}
  \label{tab:confirmatory}
\end{table*}
"""
    )


def _table_4(inputs: dict[str, Any]) -> str:
    rows: list[str] = []
    for model_index, key in enumerate(MODEL_ORDER):
        model = inputs["models"][key]
        short_name = "Qwen" if key == "qwen" else "Mistral"
        for arm_index, arm in enumerate(ARM_ORDER[1:]):
            contrast = model["paired_contrasts"][f"{arm}_minus_neutral"]
            cells = []
            for metric in ("mechanism_accuracy", "cue_following"):
                item = contrast[metric]
                cells.append(
                    f"${_signed(item['value'])}\\;[{_fmt(item['ci_low'])},{_fmt(item['ci_high'])}]$"
                )
                if item["denominator"] != 64:
                    raise ValueError(f"{key}/{arm}/{metric}: expected 64 paired seeds")
            model_cell = short_name if arm_index == 0 else ""
            rows.append("    " + " & ".join((model_cell, ARM_LABELS[arm], *cells)) + r" \\")
        if model_index == 0:
            rows.append(r"    \midrule")
    return (
        r"""
\begin{table}[t]
  \centering
  \small
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{llrr}
    \toprule
    Model & Pool & $\Delta\mathrm{Acc}_M$ & $\Delta F_C$ \\
    \midrule
"""
        + "\n".join(rows)
        + r"""
    \bottomrule
  \end{tabular}
  \caption{Paired arm-minus-neutral contrasts for the two fixed cue and mechanism
  pools, reported separately. Intervals use 10,000 paired base-seed resamples;
  every row has $n=64$ paired seeds.}
  \label{tab:confirmatory-contrasts}
\end{table}
"""
    )


def _matplotlib():
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1786492800")
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.compression": 9,
            "pdf.fonttype": 42,
            "svg.hashsalt": "epistemic-crucible-v02",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure, base: pathlib.Path) -> None:
    metadata = {
        "Creator": "Epistemic Crucible submission asset generator",
        "Producer": "matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    svg_path = base.with_suffix(".svg")
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Creator": metadata["Creator"], "Date": None},
    )
    # Matplotlib emits path-data lines with trailing spaces. Canonicalize them
    # so generated vector assets pass repository whitespace checks byte-for-byte.
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )


def _figure_1(output: pathlib.Path) -> None:
    from crucible.factorial import generate_affordance_quartet
    from crucible.viz import render_factorial_world

    plt = _matplotlib()
    figure = plt.figure(figsize=(7.0, 3.7))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.35, 0.85), wspace=0.16)
    world_ax = figure.add_subplot(grid[0, 0])
    render_factorial_world(generate_affordance_quartet(7).cell(0, 1), ax=world_ax)
    world_ax.set_title("One public factorial world")
    action_ax = figure.add_subplot(grid[0, 1])
    action_ax.axis("off")
    boxes = (
        (0.5, 0.78, "Observe", "visible color, shape,\ntexture, position"),
        (0.5, 0.48, "QUERY(tool)", "detector returns\nsignal ±"),
        (0.5, 0.18, "COMMIT(tool)", "terminal gate\napplication"),
    )
    for x, y, heading, detail in boxes:
        action_ax.text(
            x,
            y,
            f"{heading}\n{detail}",
            ha="center",
            va="center",
            transform=action_ax.transAxes,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#F5F7FA",
                "edgecolor": "#30363D",
            },
        )
    for start, end in ((0.69, 0.59), (0.39, 0.29)):
        action_ax.annotate(
            "",
            xy=(0.5, end),
            xytext=(0.5, start),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#30363D"},
        )
    action_ax.text(
        0.5,
        0.01,
        "≤3 distinct queries; exactly one commitment",
        ha="center",
        va="bottom",
        fontsize=7.5,
        transform=action_ax.transAxes,
    )
    _save_figure(figure, output / "figure1_factorial_world")
    plt.close(figure)


def _draw_design(ax, rows: int, title: str) -> None:
    from matplotlib.patches import Rectangle

    ax.set_xlim(-0.8, 2.05)
    ax.set_ylim(-0.55, rows + 0.55)
    ax.axis("off")
    ax.set_title(title, weight="bold", pad=3)
    for m in range(rows):
        y = rows - 1 - m
        ax.text(-0.10, y + 0.4, f"mechanism {m}", ha="right", va="center", fontsize=8)
        for c in range(2):
            rect = Rectangle(
                (c + 0.05, y + 0.05),
                0.9,
                0.7,
                facecolor="#F6F8FA",
                edgecolor="#30363D",
                linewidth=1.0,
            )
            ax.add_patch(rect)
            ax.text(c + 0.5, y + 0.4, rf"$m{m}\_c{c}$", ha="center", va="center")
    for c in range(2):
        ax.text(c + 0.5, rows + 0.03, f"red cue {c}", ha="center", va="bottom", fontsize=8)


def _figure_2(output: pathlib.Path) -> None:
    plt = _matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"wspace": 0.28})
    _draw_design(axes[0], 2, r"Construct validation: $2\times2$")
    _draw_design(axes[1], 3, r"Confirmatory challenge: $3\times2$")
    axes[0].text(
        0.62,
        -0.35,
        "cross hidden mechanism × visible cue",
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
    )
    axes[1].text(
        0.62,
        -0.35,
        "green tool joins the mechanism axis",
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
    )
    _save_figure(figure, output / "figure2_crossed_designs")
    plt.close(figure)


def _figure_3(output: pathlib.Path, integrity: dict[str, Any], bc: list[dict[str, Any]]) -> None:
    plt = _matplotlib()
    figure, ax = plt.subplots(figsize=(5.7, 4.3))
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xlabel(r"Mechanism responsiveness $R_M$")
    ax.set_ylabel(r"Cue susceptibility $S_C^{commit}$")
    ax.grid(color="#E2E6EA", linewidth=0.8)

    control_styles = {
        "mechanism_oracle": ("oracle", "o"),
        "detector_policy": ("detector", "s"),
        "cue_follower": ("cue follower", "D"),
        "anti_cue": ("anti-cue", "v"),
        "fixed_slot": ("fixed slot", "P"),
        "random_committer": ("uniform 3-tool", "X"),
        "focal_uniform": ("uniform focal", "^"),
    }
    for key, (label, marker) in control_styles.items():
        report = integrity["controls"][key]
        ax.scatter(
            report["mechanism_responsiveness"]["value"],
            report["cue_susceptibility"]["value"],
            marker=marker,
            s=50,
            facecolor="#B4BBC4",
            edgecolor="#343A40",
            linewidth=0.8,
            label=label,
            zorder=3,
        )

    bc_points: dict[str, tuple[float, float]] = {}
    for regime in ("cue", "mechanism"):
        reports = [run["metrics"] for run in bc if run["regime"] == regime]
        x = sum(report["mechanism_responsiveness"]["value"] for report in reports) / len(reports)
        y = sum(report["cue_susceptibility"]["value"] for report in reports) / len(reports)
        bc_points[regime] = (x, y)
    for regime, (x, y) in bc_points.items():
        color = "#C85146" if regime == "cue" else "#3274A1"
        ax.scatter(
            x,
            y,
            s=135,
            facecolor=color,
            edgecolor="white",
            linewidth=1.4,
            label=f"{regime.title()}-BC (5 seeds)",
            zorder=5,
        )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_figure(figure, output / "figure3_attribution_calibration")
    plt.close(figure)


def _errors(metric: dict[str, Any]) -> tuple[float, float]:
    value = metric["value"]
    return value - metric["ci_low"], metric["ci_high"] - value


def _figure_4(output: pathlib.Path, inputs: dict[str, Any]) -> None:
    plt = _matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    x = list(range(len(ARM_ORDER)))
    styles = {
        "mechanism_accuracy": (r"Mechanism accuracy", "#3274A1", "o"),
        "cue_following": (r"Cue following", "#D57A1F", "s"),
        "detector_query_rate": (r"Detector-query rate", "#4E9B50", "^"),
    }
    for ax, key in zip(axes, MODEL_ORDER):
        reports = inputs["models"][key]["reports"]
        for metric_name, (label, color, marker) in styles.items():
            metrics = [reports[arm][metric_name] for arm in ARM_ORDER]
            values = [metric["value"] for metric in metrics]
            errors = list(zip(*[_errors(metric) for metric in metrics]))
            ax.errorbar(
                x,
                values,
                yerr=errors,
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=4.8,
                capsize=2.2,
                label=label,
            )
        ax.set_title(inputs["models"][key]["display_name"], weight="bold")
        ax.set_xticks(x, ("N", r"$C_A$", r"$C_B$", r"$M_A$", r"$M_B$"))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color="#E2E6EA", linewidth=0.8)
        ax.set_xlabel("fixed prompt arm")
    axes[0].set_ylabel("rate")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.subplots_adjust(bottom=0.25, left=0.08, right=0.99, top=0.88, wspace=0.12)
    _save_figure(figure, output / "figure4_prompt_profiles")
    plt.close(figure)


def _refresh_parent_result_manifest(output: pathlib.Path) -> None:
    """Hash every compact result artifact once generated assets are present."""
    result_root = output.parent
    manifest_path = result_root / "artifact_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _load(manifest_path)
    manifest["files"] = {}
    for path in sorted(result_root.rglob("*")):
        if path.is_file() and path != manifest_path:
            name = path.relative_to(result_root).as_posix()
            manifest["files"][name] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build_assets(
    *,
    inputs_path: pathlib.Path,
    integrity_path: pathlib.Path,
    bc_path: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, dict[str, Any]]:
    """Build all assets and return their content manifest."""
    inputs = _load(inputs_path)
    integrity = _load(integrity_path)
    bc = _load(bc_path)
    output.mkdir(parents=True, exist_ok=True)

    _write(output / "table1_integrity.tex", _table_1(integrity))
    _write(output / "table2_bc.tex", _table_2(bc))
    _write(output / "table3_confirmatory.tex", _table_3(inputs))
    _write(output / "table4_contrasts.tex", _table_4(inputs))
    _figure_1(output)
    _figure_2(output)
    _figure_3(output, integrity, bc)
    _figure_4(output, inputs)

    manifest: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in {"asset_manifest.json", "SHA256SUMS"}:
            manifest[path.name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    (output / "asset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2-submission-assets-1",
                "inputs": {
                    "manuscript_inputs_sha256": sha256(inputs_path),
                    "integrity_report_sha256": sha256(integrity_path),
                    "factorial_bc_summary_sha256": sha256(bc_path),
                },
                "files": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    lines = [f"{item['sha256']}  {name}" for name, item in sorted(manifest.items())]
    _write(output / "SHA256SUMS", "\n".join(lines))
    _refresh_parent_result_manifest(output)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        dest="inputs_path",
        type=pathlib.Path,
        default=pathlib.Path("results/reference/v02/confirmatory/manuscript_inputs.json"),
    )
    parser.add_argument(
        "--integrity",
        dest="integrity_path",
        type=pathlib.Path,
        default=pathlib.Path("results/reference/v02/integrity_report.json"),
    )
    parser.add_argument(
        "--bc",
        dest="bc_path",
        type=pathlib.Path,
        default=pathlib.Path("results/reference/v02/factorial_bc_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/reference/v02/confirmatory/manuscript"),
    )
    args = parser.parse_args(argv)
    build_assets(**vars(args))


if __name__ == "__main__":
    main()
