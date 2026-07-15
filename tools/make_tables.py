"""
tools/make_tables.py — regenerate dissertation tables straight from the CSVs.

The write-up's result tables MUST match the committed CSVs (an earlier audit
found RESEARCH_NOTES quoting stale, pre-tuning numbers that contradicted the
data). This script reads whatever is in research_results/ and emits clean
Markdown so no figure is ever hand-transcribed. Re-run it after every study.

Reads (any that exist):
    summary_multiseed.csv, pvalues.csv          (ablation)
    generalist_summary.csv, generalist_pvalues.csv
    baselines_summary.csv, baselines_pvalues.csv
Writes:
    research_results/RESULTS_TABLES.md

Usage:  python tools/make_tables.py
"""

import os
import sys

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJ)

import pandas as pd                     # noqa: E402
import config as cfg                    # noqa: E402

RD = cfg.RESULTS_DIR


def _load(name):
    p = os.path.join(RD, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _fmt(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except (ValueError, TypeError):
        return str(x)


def summary_block(df, title, order=None):
    """Pivot a *_summary.csv (Configuration/Agent × Metric, Mean±CI) to Markdown."""
    if df is None or df.empty:
        return f"### {title}\n\n_(CSV not found — run the study first.)_\n"
    key = "Configuration" if "Configuration" in df.columns else "Agent"
    group = "Experiment" if "Experiment" in df.columns else "Task"
    out = [f"### {title}\n"]
    for gval, gdf in df.groupby(group, sort=False):
        metrics = list(dict.fromkeys(gdf["Metric"]))
        rows = list(dict.fromkeys(gdf[key]))
        if order:
            rows = [r for r in order if r in rows] + [r for r in rows if r not in (order or [])]
        out.append(f"\n**{gval}**  _(mean ± 95% CI, N={int(gdf['N_seeds'].max())} seeds)_\n")
        out.append("| " + key + " | " + " | ".join(metrics) + " |")
        out.append("|" + "---|" * (len(metrics) + 1))
        for r in rows:
            cells = []
            for m in metrics:
                sub = gdf[(gdf[key] == r) & (gdf["Metric"] == m)]
                if sub.empty:
                    cells.append("—")
                else:
                    mean = _fmt(sub["Mean"].iloc[0])
                    ci = _fmt(sub["CI95"].iloc[0])
                    cells.append(f"{mean} ± {ci}")
            out.append(f"| {r} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def pvalue_block(df, title):
    """Render a *_pvalues.csv (paired Full-vs-X tests) as Markdown."""
    if df is None or df.empty:
        return f"### {title}\n\n_(CSV not found — run the study first.)_\n"
    # Choose a compact, robust column set (some files use different names).
    prefer = ["Experiment", "Task", "Comparison", "Metric", "Full_Mean",
              "Other_Mean", "Full_mean", "Spec_mean", "Base_mean",
              "p_value", "p_holm", "p_wilcoxon", "p_wilcoxon_holm",
              "sig_holm_0.05", "sig_wilcoxon_holm_0.05", "Full_Better",
              "Full_better", "N_pairs", "N"]
    cols = [c for c in prefer if c in df.columns]
    out = [f"### {title}\n", "| " + " | ".join(cols) + " |",
           "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    parts = ["# Results tables (auto-generated from research_results/*.csv)\n",
             "_Regenerate with `python tools/make_tables.py`. Do not hand-edit "
             "numbers — edit the study and re-run._\n",
             "\n---\n\n## 1. Ablation study (same-architecture, single-variable)\n",
             summary_block(_load("summary_multiseed.csv"),
                           "Per-config metrics",
                           order=list(cfg.ABLATION_CONFIGS.keys())),
             pvalue_block(_load("pvalues.csv"),
                          "Paired tests — Full vs each config (t + Wilcoxon, Holm)"),
             "\n---\n\n## 2. Generalist vs specialists (integration novelty)\n",
             summary_block(_load("generalist_summary.csv"), "Per-agent metrics",
                           order=list(cfg.GENERALIST_CONFIGS.keys())),
             pvalue_block(_load("generalist_pvalues.csv"),
                          "Paired tests — Full vs each specialist"),
             "\n---\n\n## 3. Fair baselines (beats STANDARD RL?)\n",
             summary_block(_load("baselines_summary.csv"), "Per-agent metrics",
                           order=list(cfg.BASELINE_CONFIGS.keys())),
             pvalue_block(_load("baselines_pvalues.csv"),
                          "Paired tests — Full vs each standard-DQN baseline")]
    text = "\n".join(parts)
    out_path = os.path.join(RD, "RESULTS_TABLES.md")
    os.makedirs(RD, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
