"""Head-to-head analysis with error bars + the colour-shortcut reliance metric.

Loads one or more JSONL traces (symbolic, neural, and LLM agents share the same
schema), computes per-(agent, family, split) TSR with bootstrap CIs, and the
behavioural colour-reliance metric on the affordance family — which is
measurable even when task success is low. Writes a markdown report + JSON.

Usage:
    python experiments/analyze_results.py --traces results/*.jsonl results/llm_*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import sys

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.grammar import TaskFamily, generate_task  # noqa: E402
from crucible.metrics import filter_records, load_traces, tsr_with_ci  # noqa: E402
from crucible.splits import SplitLabel  # noqa: E402
from crucible.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


def _red_tool_and_gate(seed: int, split: str) -> tuple[set[str], str | None]:
    """Real IDs of red-coloured tools and the gate for an affordance instance."""
    spec = generate_task(TaskFamily.AFFORDANCE, seed=seed, split=SplitLabel(split))
    red = {
        o.obj_id
        for o in spec.object_specs
        if o.obj_type.value == "tool" and o.color.value == "red"
    }
    gate = next((o.obj_id for o in spec.object_specs if o.obj_type.value == "gate"), None)
    return red, gate


def colour_reliance(steps: list[dict], agent: str, split: str) -> dict:
    """Fraction of affordance episodes whose first apply-to-gate used a RED tool.

    On TRAIN red is the conductive tool (reliance = correct); on TEST red is a
    decoy (reliance = shortcut error). A train/test reliance gap, or high TEST
    reliance, indicates colour-prior dependence — measurable even at TSR 0.
    """
    aff = [s for s in filter_records(steps, family="affordance", agent=agent, split=split)]
    by_ep: dict[tuple, list[dict]] = {}
    for s in aff:
        by_ep.setdefault((s["seed"], s["episode"]), []).append(s)
    applied_any, applied_red = 0, 0
    for (seed, _ep), ep_steps in by_ep.items():
        red, gate = _red_tool_and_gate(seed, split)
        if gate is None:
            continue
        for s in sorted(ep_steps, key=lambda r: r["step"]):
            act = s.get("action", {})
            if act.get("kind") == "apply" and act.get("args", {}).get("target_id") == gate:
                applied_any += 1
                if act.get("args", {}).get("tool_id") in red:
                    applied_red += 1
                break
    return {
        "reliance": round(applied_red / applied_any, 4) if applied_any else None,
        "n_apply_to_gate": applied_any,
        "n_episodes": len(by_ep),
    }


_DOSE_RE = re.compile(r"^(cue|mechanistic|anticue)_k(\d+)$")
_ORACLE_LEVELS = ("none", "intervention", "property", "rule")


def _single_agent(outcomes: list[dict]) -> str | None:
    ags = {o["agent"] for o in outcomes if "agent" in o}
    return next(iter(ags)) if len(ags) == 1 else (sorted(ags)[0] if ags else None)


def dose_response(dose_dir: pathlib.Path, family: str = "affordance") -> dict:
    """Reliance/TSR vs demo count K for each demo mode (the dose-response probe).

    Scans ``dose_dir`` for ``k0`` (the zero-shot baseline, shared K=0 point) and
    ``<mode>_k<K>`` subdirectories (mode in cue/mechanistic/anticue), each holding
    the traces for one condition. For each condition it reports the colour-reliance
    on TEST (where colour is a decoy: higher = more shortcut use) and on TRAIN,
    plus task success. The headline curve is TEST reliance against K per mode:
    cue should rise, mechanistic stay flat, anti-cue fall.
    """
    def _one(sub: pathlib.Path) -> dict:
        paths = sorted(str(p) for p in sub.glob("*.jsonl"))
        if not paths:
            return {}
        steps, outcomes = load_traces(paths)
        ag = _single_agent(outcomes)
        return {
            "reliance_test": colour_reliance(steps, ag, "test"),
            "reliance_train": colour_reliance(steps, ag, "train"),
            "tsr_test": tsr_with_ci(outcomes, family=family, agent=ag, split="test"),
            "tsr_train": tsr_with_ci(outcomes, family=family, agent=ag, split="train"),
        }

    base = _one(dose_dir / "k0") if (dose_dir / "k0").is_dir() else {}
    modes: dict[str, list[dict]] = {}
    for sub in sorted(p for p in dose_dir.iterdir() if p.is_dir()):
        m = _DOSE_RE.match(sub.name)
        if not m:
            continue
        mode, k = m.group(1), int(m.group(2))
        rec = _one(sub)
        if rec:
            modes.setdefault(mode, []).append({"k": k, **rec})
    for rows in modes.values():
        if base:
            rows.append({"k": 0, **base})
        rows.sort(key=lambda r: r["k"])
    return {"family": family, "baseline_k0": base, "modes": modes}


def oracle_ladder(oracle_dir: pathlib.Path) -> dict:
    """Task success by oracle level (the failure-localization ladder).

    Each subdirectory of ``oracle_dir`` is one oracle condition; its level is read
    from the first known keyword (none/intervention/property/rule) in the name.
    Returns ``level -> agent -> family -> {train, test, gap}`` so we can see where
    on the ladder each model recovers — naming the intervention, the property, or
    the full rule.
    """
    out: dict = {}
    for sub in sorted(p for p in oracle_dir.iterdir() if p.is_dir()):
        level = next((lv for lv in _ORACLE_LEVELS if lv in sub.name), None)
        if level is None:
            continue
        paths = sorted(str(p) for p in sub.glob("*.jsonl"))
        if not paths:
            continue
        _steps, outcomes = load_traces(paths)
        agents = sorted({o["agent"] for o in outcomes if "agent" in o})
        fams = sorted({o["family"] for o in outcomes if "family" in o})
        lvl: dict = out.setdefault(level, {})
        for ag in agents:
            per = lvl.setdefault(ag, {})
            for fam in fams:
                tr = tsr_with_ci(outcomes, family=fam, agent=ag, split="train")
                te = tsr_with_ci(outcomes, family=fam, agent=ag, split="test")
                if tr["n"] or te["n"]:
                    per[fam] = {"train": tr, "test": te,
                                "gap": round(tr["mean"] - te["mean"], 4)}
    return out


def analyze(
    trace_paths: list[str],
    out_dir: pathlib.Path,
    fewshot_paths: list[str] | None = None,
    fewshot_split: str = "test",
    dose_dir: pathlib.Path | None = None,
    oracle_dir: pathlib.Path | None = None,
) -> dict:
    steps, outcomes = load_traces(trace_paths)
    agents = sorted({o["agent"] for o in outcomes if "agent" in o})
    families = sorted({o["family"] for o in outcomes if "family" in o})

    table: dict = {"families": families, "agents": {}}
    for ag in agents:
        per_family = {}
        for fam in families:
            tr = tsr_with_ci(outcomes, family=fam, agent=ag, split="train")
            te = tsr_with_ci(outcomes, family=fam, agent=ag, split="test")
            gap = round(tr["mean"] - te["mean"], 4)
            per_family[fam] = {"train": tr, "test": te, "gap": gap}
        reliance = {
            "train": colour_reliance(steps, ag, "train"),
            "test": colour_reliance(steps, ag, "test"),
        }
        table["agents"][ag] = {
            "overall": tsr_with_ci(outcomes, agent=ag),
            "tsr": per_family,
            "affordance_colour_reliance": reliance,
        }

    # Few-shot in-context shortcut probe: compare zero-shot vs primed colour
    # reliance on the held-out split (only for agents present in both sets).
    if fewshot_paths:
        fs_steps, fs_outcomes = load_traces(fewshot_paths)
        fs_agents = sorted({o["agent"] for o in fs_outcomes if "agent" in o})
        table["fewshot_reliance"] = {}
        for ag in fs_agents:
            zero = colour_reliance(steps, ag, fewshot_split)
            few = colour_reliance(fs_steps, ag, fewshot_split)
            table["fewshot_reliance"][ag] = {
                "zero_shot": zero["reliance"],
                "few_shot": few["reliance"],
                "delta": (
                    round(few["reliance"] - zero["reliance"], 4)
                    if (few["reliance"] is not None and zero["reliance"] is not None)
                    else None
                ),
                "n_zero": zero["n_apply_to_gate"],
                "n_few": few["n_apply_to_gate"],
            }

    # Dose-response shortcut probe and oracle ablation ladder (optional inputs).
    if dose_dir and dose_dir.is_dir():
        table["dose_response"] = dose_response(dose_dir)
    if oracle_dir and oracle_dir.is_dir():
        table["oracle_ladder"] = oracle_ladder(oracle_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(table, indent=2))
    md = _render_markdown(table)
    (out_dir / "analysis.md").write_text(md)
    try:
        _make_figures(table, out_dir)
    except Exception as exc:  # matplotlib optional; never fail the analysis
        _log.warning("figure generation skipped: %s", exc)
    _log.info("wrote analysis.json and analysis.md to %s", out_dir)
    print(md)
    return table


def _make_figures(table: dict, out_dir: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Publication-quality defaults: 300 dpi and fonts set large enough to stay
    # legible after the figures are scaled to the column width in the paper.
    plt.rcParams.update({
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "savefig.bbox": "tight",
    })

    agents = list(table["agents"])
    aff = [table["agents"][a]["tsr"].get("affordance", {}) for a in agents]
    train = [d.get("train", {}).get("mean", 0.0) for d in aff]
    test = [d.get("test", {}).get("mean", 0.0) for d in aff]
    tr_err = [
        [d.get("train", {}).get("mean", 0) - d.get("train", {}).get("lo", 0) for d in aff],
        [d.get("train", {}).get("hi", 0) - d.get("train", {}).get("mean", 0) for d in aff],
    ]
    te_err = [
        [d.get("test", {}).get("mean", 0) - d.get("test", {}).get("lo", 0) for d in aff],
        [d.get("test", {}).get("hi", 0) - d.get("test", {}).get("mean", 0) for d in aff],
    ]
    x = range(len(agents))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(agents) * 1.4), 4.5))
    ax.bar([i - w / 2 for i in x], train, w, yerr=tr_err, capsize=3, label="train", color="#4c72b0")
    ax.bar([i + w / 2 for i in x], test, w, yerr=te_err, capsize=3, label="test", color="#dd8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels([a.replace("claude-", "").replace("-20251001", "") for a in agents],
                       rotation=30, ha="right")
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Affordance: train vs test TSR (decision-focused, 95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tsr_train_vs_test.png", dpi=300)
    plt.close(fig)

    # Dose-response figure: TEST colour reliance vs K, one line per demo mode.
    dose = table.get("dose_response")
    if dose and dose.get("modes"):
        colours = {"cue": "#c44e52", "mechanistic": "#55a868", "anticue": "#4c72b0"}
        markers = {"cue": "o", "mechanistic": "s", "anticue": "^"}
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for mode in ("cue", "mechanistic", "anticue"):
            rows = dose["modes"].get(mode)
            if not rows:
                continue
            xs = [r["k"] for r in rows if r["reliance_test"]["reliance"] is not None]
            ys = [r["reliance_test"]["reliance"] for r in rows
                  if r["reliance_test"]["reliance"] is not None]
            ax.plot(xs, ys, marker=markers[mode], color=colours[mode],
                    label=mode, linewidth=2, markersize=7)
        ax.set_xlabel("Number of in-context demonstrations (K)")
        ax.set_ylabel("Test-split colour reliance")
        ax.set_title("Dose-response: in-context shortcut acquisition")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(title="demo mode")
        fig.tight_layout()
        fig.savefig(out_dir / "dose_response.png", dpi=300)
        plt.close(fig)


def _fmt(ci: dict) -> str:
    return f"{ci['mean']:.2f} [{ci['lo']:.2f},{ci['hi']:.2f}]" if ci["n"] else "—"


def _render_markdown(table: dict) -> str:
    lines = ["# Head-to-head results\n"]

    # Overall capability gradient (all families, both splits), sorted desc.
    lines.append("## Capability gradient — overall TSR (all families, decision-focused)\n")
    lines.append("| Agent | Overall TSR [95% CI] | n |")
    lines.append("| --- | --- | --- |")
    ranked = sorted(
        table["agents"].items(), key=lambda kv: kv[1]["overall"]["mean"], reverse=True
    )
    for ag, data in ranked:
        o = data["overall"]
        lines.append(f"| {ag} | {o['mean']:.3f} [{o['lo']:.2f},{o['hi']:.2f}] | {o['n']} |")

    # Per-family TSR (train -> test).
    fams = table["families"]
    lines.append("\n## Per-family TSR (train → test)\n")
    lines.append("| Agent | " + " | ".join(fams) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in fams) + " |")
    for ag, data in ranked:
        cells = []
        for fam in fams:
            d = data["tsr"].get(fam, {})
            tr = d.get("train", {}).get("mean")
            te = d.get("test", {}).get("mean")
            cells.append(f"{tr:.2f}→{te:.2f}" if tr is not None else "—")
        lines.append(f"| {ag} | " + " | ".join(cells) + " |")

    lines.append("\n## Affordance: TSR (mean [95% CI]) and colour reliance\n")
    lines.append("| Agent | Train TSR | Test TSR | Gap | Train red-apply | Test red-apply |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for ag, data in table["agents"].items():
        aff = data["tsr"].get("affordance", {})
        rel = data["affordance_colour_reliance"]
        tr_rel = rel["train"]["reliance"]
        te_rel = rel["test"]["reliance"]
        lines.append(
            f"| {ag} | {_fmt(aff.get('train', {'n': 0}))} | {_fmt(aff.get('test', {'n': 0}))} "
            f"| {aff.get('gap', 0):+.2f} "
            f"| {tr_rel if tr_rel is not None else '—'} "
            f"| {te_rel if te_rel is not None else '—'} |"
        )
    lines.append("\n*Red-apply = fraction of episodes whose first apply-to-gate used a red tool. "
                 "On TRAIN red is conductive (correct); on TEST red is a decoy (shortcut).*\n")

    # Few-shot in-context shortcut probe.
    fewshot = table.get("fewshot_reliance")
    if fewshot:
        lines.append("\n## Few-shot in-context shortcut probe (affordance, held-out split)\n")
        lines.append(
            "K=3 solved TRAIN demos (conductive tool is RED), then evaluated on TEST worlds "
            "where colour is decorrelated. Higher = more reliance on the spurious colour rule.\n"
        )
        lines.append("| Agent | Zero-shot red-apply | Few-shot red-apply | Δ |")
        lines.append("| --- | --- | --- | --- |")
        for ag, d in sorted(fewshot.items()):
            zs = d["zero_shot"] if d["zero_shot"] is not None else "—"
            fs = d["few_shot"] if d["few_shot"] is not None else "—"
            dl = f"{d['delta']:+.2f}" if d["delta"] is not None else "—"
            lines.append(f"| {ag} | {zs} | {fs} | {dl} |")

    # Dose-response: TEST colour reliance vs K, per demo mode.
    dose = table.get("dose_response")
    if dose and dose.get("modes"):
        lines.append("\n## Dose-response: in-context shortcut vs demo count K\n")
        lines.append(
            "TEST-split colour reliance (fraction of first apply-to-gate using a red "
            "decoy tool) after K demonstrations. *cue* demos make red the conductive "
            "tool; *mechanistic* demos show discovery with no colour regularity; "
            "*anti-cue* demos vary the correct tool's colour.\n"
        )
        for mode in ("cue", "mechanistic", "anticue"):
            rows = dose["modes"].get(mode)
            if not rows:
                continue
            lines.append(f"\n**{mode}**\n")
            lines.append("| K | Test reliance | Test TSR | Train reliance | n apply (test) |")
            lines.append("| --- | --- | --- | --- | --- |")
            for r in rows:
                rt = r["reliance_test"]["reliance"]
                rtr = r["reliance_train"]["reliance"]
                tt = r["tsr_test"]
                lines.append(
                    f"| {r['k']} | {rt if rt is not None else '—'} "
                    f"| {tt['mean']:.2f} "
                    f"| {rtr if rtr is not None else '—'} "
                    f"| {r['reliance_test']['n_apply_to_gate']} |"
                )

    # Oracle ladder: TSR by oracle level.
    oracle = table.get("oracle_ladder")
    if oracle:
        lines.append("\n## Oracle ladder: where failure is localized\n")
        lines.append(
            "Task success when a labelled hint is injected: *intervention* names the "
            "operative action, *property* names the tool that works, *rule* states the "
            "full local rule. Train→test per family.\n"
        )
        levels = [lv for lv in _ORACLE_LEVELS if lv in oracle]
        agents = sorted({a for lv in oracle.values() for a in lv})
        fams = sorted({f for lv in oracle.values() for a in lv.values() for f in a})
        for fam in fams:
            lines.append(f"\n**{fam}**\n")
            lines.append("| Agent | " + " | ".join(levels) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in levels) + " |")
            for ag in agents:
                cells = []
                for lv in levels:
                    d = oracle.get(lv, {}).get(ag, {}).get(fam)
                    cells.append(
                        f"{d['train']['mean']:.2f}→{d['test']['mean']:.2f}" if d else "—"
                    )
                lines.append(f"| {ag} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        out.extend(sorted(glob.glob(pat)) or ([pat] if pathlib.Path(pat).exists() else []))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Head-to-head analysis with error bars.")
    p.add_argument("--traces", nargs="+", default=["results/*.jsonl"])
    p.add_argument(
        "--fewshot-traces", nargs="+", help="Few-shot probe traces for the reliance table"
    )
    p.add_argument("--fewshot-split", default="test")
    p.add_argument("--dose-dir", help="Directory of dose-response condition subdirs")
    p.add_argument("--oracle-dir", help="Directory of oracle-level condition subdirs")
    p.add_argument("--output-dir", default="results")
    args = p.parse_args(argv)
    paths = _expand(args.traces)
    if not paths:
        _log.error("no trace files matched")
        return
    fewshot = _expand(args.fewshot_traces) if args.fewshot_traces else None
    analyze(paths, pathlib.Path(args.output_dir), fewshot_paths=fewshot,
            fewshot_split=args.fewshot_split,
            dose_dir=pathlib.Path(args.dose_dir) if args.dose_dir else None,
            oracle_dir=pathlib.Path(args.oracle_dir) if args.oracle_dir else None)


if __name__ == "__main__":
    main()
