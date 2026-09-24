"""Results reporting: markdown tables generated from ``reports/metrics.json``.

The README's *Results* section is filled from these generated tables (between
``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->``), so it only ever shows
numbers produced by an actual training run.

Usage::

    python -m src.evaluation.report                # print + write reports/results.md
    python -m src.evaluation.report --update-readme
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, get_settings

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
HORIZON_LABELS = {1: "1 hour", 6: "6 hours", 24: "24 hours", 168: "7 days (168 h)"}


def _fmt(v: Any, digits: int = 2) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _table(metrics: dict[str, dict], selected: str) -> list[str]:
    lines = [
        "| Model | MAE (MW) | RMSE (MW) | MAPE (%) | sMAPE (%) | Bias (MW) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in sorted(metrics.items(), key=lambda kv: kv[1]["mae"]):
        label = f"**{name}** ✅" if name == selected else name
        lines.append(
            f"| {label} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {_fmt(m['mape'])} | "
            f"{_fmt(m['smape'])} | {_fmt(m['bias'])} |"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    ds = report["dataset"]
    test = report["splits"]["test"]
    folds = report["splits"]["folds"]
    out = [
        f"_Generated automatically from `reports/metrics.json` — training run of "
        f"{report['trained_at'][:19]} UTC, dataset `{ds['source']}` version `{ds['version']}` "
        f"({ds['rows']:,} hourly rows, {ds['start'][:10]} → {ds['end'][:10]})._",
        "",
    ]
    if ds.get("is_synthetic"):
        out += [
            "> ⚠️ These metrics are measured on **synthetic data**. They validate the pipeline and the "
            "methodology; they are **not** an estimate of accuracy on real Nouakchott demand.",
            "",
        ]
    out += [
        f"- **Cross-validation (model selection)**: {len(folds)} expanding-window folds of "
        f"{report['config']['split']['fold_days']} days, {folds[0]['start'][:10]} → {folds[-1]['end'][:10]}.",
        f"- **Held-out test (reporting only)**: {test['start'][:10]} → {test['end'][:10]}. "
        "Models are refitted on all development data, then scored once.",
        "- Metrics are *H-step-ahead*: each target hour is forecast from information available H hours earlier.",
        "",
        "### Selected model per horizon",
        "",
        "| Horizon | Selected (CV MAE) | Test MAE (MW) | Test RMSE (MW) | Test MAPE (%) | "
        "Best baseline test MAE | 90 % interval coverage (test) | Mean interval width (MW) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    baselines = ("Naive", "Seasonal Naive", "Weekly Seasonal Naive")
    for h, r in sorted(report["horizons"].items(), key=lambda kv: int(kv[0])):
        sel = r["selected_model"]
        t = r["test"][sel]
        base = min((m["mae"], n) for n, m in r["test"].items() if n in baselines)
        iv = r["intervals"]["test"]
        out.append(
            f"| {HORIZON_LABELS.get(int(h), h + ' h')} | {sel} | {_fmt(t['mae'])} | {_fmt(t['rmse'])} | "
            f"{_fmt(t['mape'])} | {_fmt(base[0])} ({base[1]}) | {_fmt(iv['coverage'] * 100, 1)} % | "
            f"{_fmt(iv['mean_width_mw'], 1)} |"
        )

    for h, r in sorted(report["horizons"].items(), key=lambda kv: int(kv[0])):
        sel = r["selected_model"]
        pk = r["peaks"]["test"]
        out += [
            "",
            f"### Horizon: {HORIZON_LABELS.get(int(h), h + ' h')}",
            "",
            "**Cross-validation (pooled out-of-sample, used for selection)**",
            "",
            *_table(r["cv"], sel),
            "",
            "**Held-out test period**",
            "",
            *_table(r["test"], sel),
            "",
            f"Peak alerts on test (threshold P{int(r['peaks']['percentile'] * 100)} of the trailing "
            f"{r['peaks']['window_days']} days = {_fmt(pk['threshold_mw'], 1)} MW, "
            f"{pk['actual_peak_hours']} actual peak hours): "
            f"HIGH precision {_fmt(pk['high_precision'] * 100, 0)} % / recall {_fmt(pk['high_recall'] * 100, 0)} %; "
            f"HIGH+MEDIUM precision {_fmt(pk['high_or_medium_precision'] * 100, 0)} % / "
            f"recall {_fmt(pk['high_or_medium_recall'] * 100, 0)} %.",
        ]
    return "\n".join(out) + "\n"


def write_markdown_report(report: dict[str, Any], path: Path) -> Path:
    path.write_text(render_markdown(report))
    return path


def update_readme(markdown: str, readme: Path = PROJECT_ROOT / "README.md") -> bool:
    """Replace the block between the RESULTS markers. Returns True if the README changed."""
    text = readme.read_text()
    if START not in text or END not in text:
        raise ValueError("README has no RESULTS markers")
    new = re.sub(
        re.escape(START) + r".*?" + re.escape(END),
        lambda _: f"{START}\n{markdown}{END}",
        text,
        flags=re.S,
    )
    if new != text:
        readme.write_text(new)
    return new != text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render training results as markdown.")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args(argv)
    reports_dir = get_settings().reports_dir
    metrics_path = reports_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit("No reports/metrics.json yet: run `python -m src.training.train` first.")
    report = json.loads(metrics_path.read_text())
    md = render_markdown(report)
    write_markdown_report(report, reports_dir / "results.md")
    print(md)
    if args.update_readme:
        print("README updated." if update_readme(md) else "README already up to date.")


if __name__ == "__main__":
    main()
