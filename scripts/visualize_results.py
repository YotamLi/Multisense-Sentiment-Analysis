"""
Generate publication-ready evaluation report with charts and tables.

Reads evaluation_results.json and produces:
  1. A console-printed summary table
  2. PNG charts saved to evaluation_figures/
  3. An all-in-one HTML report (evaluation_report.html)

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --input evaluation_results.json
"""

from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


COLORS = {
    "bert": "#2563EB",
    "vader": "#F59E0B",
    "nb": "#EF4444",
    "bert_single": "#8B5CF6",
    "accent": "#10B981",
}

TASK_DISPLAY = {
    "sentiment": "Sentiment",
    "emotion": "Emotion",
    "intensity": "Intensity",
    "topic": "Topic",
}


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Console table ────────────────────────────────────────────────────────

def print_summary_table(data: dict):
    bert = data.get("multi_head_bert", {})
    vader = data.get("vader_baseline", {})
    nb = data.get("naive_bayes_baseline", {})
    bert_single = data.get("bert_sentiment", {})

    sep = "=" * 78
    thin = "-" * 78
    print(f"\n{sep}")
    print("  EVALUATION REPORT — Multi-Task Sentiment Analysis (Held-Out Test Set)")
    print(sep)

    # Model comparison on sentiment
    print("\n  [1] MODEL COMPARISON — Sentiment Polarity (test.csv)")
    print(thin)
    print(f"  {'Model':<28} {'Accuracy':>10} {'Macro F1':>10} {'W-F1':>10} {'Samples':>10}")
    print(thin)
    rows = [
        ("Multi-Head BERT", bert.get("sentiment", {})),
        ("BERT-base (single head)", bert_single),
        ("VADER (rule-based)", vader),
        ("Naive Bayes (TF-IDF)", nb),
    ]
    for name, m in rows:
        if m:
            print(
                f"  {name:<28} {m.get('accuracy', 0):>10.4f} "
                f"{m.get('macro_f1', 0):>10.4f} "
                f"{m.get('weighted_f1', 0):>10.4f} "
                f"{m.get('num_evaluated', 0):>10d}"
            )
    print()

    # BERT per-task breakdown
    print(f"  [2] MULTI-HEAD BERT — Per-Task Breakdown")
    print(thin)
    print(
        f"  {'Task':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
        f"{'Macro F1':>10} {'W-F1':>10} {'Samples':>10}"
    )
    print(thin)
    for task_key, display in TASK_DISPLAY.items():
        m = bert.get(task_key, {})
        if m and isinstance(m, dict) and "accuracy" in m:
            print(
                f"  {display:<15} {m['accuracy']:>10.4f} "
                f"{m.get('macro_precision', 0):>10.4f} "
                f"{m.get('macro_recall', 0):>10.4f} "
                f"{m.get('macro_f1', 0):>10.4f} "
                f"{m.get('weighted_f1', 0):>10.4f} "
                f"{m.get('num_evaluated', 0):>10d}"
            )
    print()

    # Per-class detail for each task
    print(f"  [3] PER-CLASS DETAIL")
    print(thin)
    for task_key, display in TASK_DISPLAY.items():
        m = bert.get(task_key, {})
        if not m or not isinstance(m, dict):
            continue
        report = m.get("classification_report", {})
        classes = [
            k for k in report
            if k not in ("accuracy", "macro avg", "weighted avg")
        ]
        if not classes:
            continue
        print(f"\n  {display}:")
        print(f"  {'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        for cls in classes:
            r = report[cls]
            print(
                f"  {cls:<16} {r['precision']:>10.4f} "
                f"{r['recall']:>10.4f} {r['f1-score']:>10.4f} "
                f"{int(r['support']):>10d}"
            )
    print(f"\n{sep}\n")


# ── Chart generators ─────────────────────────────────────────────────────

def _style_ax(ax, title: str, ylabel: str = ""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)


def plot_model_comparison(data: dict, out_dir: Path):
    """Bar chart comparing all models on sentiment."""
    bert_sent = data.get("multi_head_bert", {}).get("sentiment", {})
    vader = data.get("vader_baseline", {})
    nb = data.get("naive_bayes_baseline", {})
    bert_single = data.get("bert_sentiment", {})

    models = [
        "Multi-Head\nBERT",
        "BERT-base\n(single head)",
        "VADER\n(rule-based)",
        "Naive Bayes\n(TF-IDF)",
    ]
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    metric_labels = ["Accuracy", "Macro F1", "Weighted F1"]
    sources = [bert_sent, bert_single, vader, nb]

    x = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [s.get(metric, 0) for s in sources]
        bars = ax.bar(x + i * width - width, vals, width, label=label, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{v:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style_ax(ax, "Model Comparison — Sentiment Polarity (Test Set)", "Score")

    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bert_tasks(data: dict, out_dir: Path):
    """Grouped bar chart of BERT performance across 4 tasks."""
    bert = data.get("multi_head_bert", {})
    tasks = list(TASK_DISPLAY.keys())
    displays = [TASK_DISPLAY[t] for t in tasks]

    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    labels = ["Accuracy", "Macro F1", "Weighted F1"]
    colors = [COLORS["bert"], COLORS["accent"], "#8B5CF6"]

    x = np.arange(len(tasks))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [bert.get(t, {}).get(metric, 0) for t in tasks]
        bars = ax.bar(x + i * width - width, vals, width, label=label, color=color, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{v:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(displays, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style_ax(ax, "Multi-Head BERT — Per-Task Performance (Test Set)", "Score")

    fig.tight_layout()
    fig.savefig(out_dir / "bert_per_task.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(data: dict, out_dir: Path):
    """Heatmap confusion matrices for each BERT task."""
    bert = data.get("multi_head_bert", {})

    label_map = {
        "sentiment": ["positive", "negative", "neutral"],
        "emotion": ["joy", "anger", "sadness", "fear", "surprise", "disgust"],
        "intensity": ["strong", "medium", "weak"],
        "topic": [
            "arts_&_culture", "business", "pop_culture",
            "daily_life", "sports_&_gaming", "sci_&_tech",
        ],
    }

    tasks_with_cm = [
        (task_key, display)
        for task_key, display in TASK_DISPLAY.items()
        if task_key in bert and "confusion_matrix" in bert.get(task_key, {})
    ]
    if not tasks_with_cm:
        return

    n_tasks = len(tasks_with_cm)
    ncols = min(n_tasks, 2)
    nrows = (n_tasks + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 6.5 * nrows),
    )
    if n_tasks == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (task_key, display) in enumerate(tasks_with_cm):
        ax = axes[idx]
        cm_arr = np.array(bert[task_key]["confusion_matrix"])
        labels = label_map.get(task_key, [str(i) for i in range(len(cm_arr))])
        if len(labels) > len(cm_arr):
            labels = labels[:len(cm_arr)]

        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_arr.astype(float) / row_sums

        annot_labels = np.empty_like(cm_arr, dtype=object)
        for r in range(cm_arr.shape[0]):
            for c in range(cm_arr.shape[1]):
                annot_labels[r, c] = f"{cm_arr[r, c]}\n({cm_norm[r, c]:.0%})"

        if HAS_SNS:
            sns.heatmap(
                cm_norm, annot=annot_labels, fmt="", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar=True, linewidths=0.8,
                annot_kws={"size": 11, "fontweight": "bold"},
                vmin=0, vmax=1,
            )
        else:
            ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            for r in range(cm_arr.shape[0]):
                for c in range(cm_arr.shape[1]):
                    color = "white" if cm_norm[r, c] > 0.5 else "black"
                    ax.text(c, r, annot_labels[r, c], ha="center", va="center",
                            fontsize=11, fontweight="bold", color=color)

        ax.set_title(display, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.tick_params(labelsize=10)

    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Confusion Matrices — Multi-Head BERT (Test Set)",
        fontsize=16, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(data: dict, out_dir: Path):
    """Horizontal bar chart of per-class F1 for each task."""
    bert = data.get("multi_head_bert", {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    task_list = list(TASK_DISPLAY.items())
    palette = ["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE"]

    for i, (task_key, display) in enumerate(task_list):
        ax = axes[i // 2][i % 2]
        m = bert.get(task_key, {})
        report = m.get("classification_report", {})
        classes = [
            k for k in report
            if k not in ("accuracy", "macro avg", "weighted avg")
        ]
        if not classes:
            ax.set_visible(False)
            continue

        f1_scores = [report[c]["f1-score"] for c in classes]
        supports = [int(report[c]["support"]) for c in classes]
        y_pos = np.arange(len(classes))
        bar_colors = palette[:len(classes)]

        bars = ax.barh(y_pos, f1_scores, color=bar_colors, alpha=0.88, height=0.6)

        for bar, f1, sup in zip(bars, f1_scores, supports):
            ax.text(
                bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{f1:.1%}  (n={sup})",
                va="center", fontsize=9, fontweight="bold",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlim(0, 1.15)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(display, fontsize=12, fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle(
        "Per-Class F1 Score — Multi-Head BERT (Test Set)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "per_class_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── HTML report ──────────────────────────────────────────────────────────

def generate_html_report(data: dict, fig_dir: Path, out_path: Path):
    bert = data.get("multi_head_bert", {})
    vader = data.get("vader_baseline", {})
    nb = data.get("naive_bayes_baseline", {})
    bert_single = data.get("bert_sentiment", {})

    def metric_cell(val, best=False):
        bold = " style=\"font-weight:700\"" if best else ""
        return f'<td{bold}>{val:.4f}</td>'

    # Sentiment comparison rows
    sent_rows = ""
    models = [
        ("Multi-Head BERT", bert.get("sentiment", {}), True),
        ("BERT-base (single head)", bert_single, False),
        ("VADER (rule-based)", vader, False),
        ("Naive Bayes (TF-IDF)", nb, False),
    ]
    for name, m, is_best in models:
        if not m:
            continue
        sent_rows += f"""<tr>
            <td style="font-weight:{'700' if is_best else '400'}">{name}</td>
            {metric_cell(m.get('accuracy', 0), is_best)}
            {metric_cell(m.get('macro_f1', 0), is_best)}
            {metric_cell(m.get('weighted_f1', 0), is_best)}
            <td>{m.get('num_evaluated', 0):,}</td>
        </tr>"""

    # BERT per-task rows
    task_rows = ""
    for task_key, display in TASK_DISPLAY.items():
        m = bert.get(task_key, {})
        if not m or not isinstance(m, dict) or "accuracy" not in m:
            continue
        task_rows += f"""<tr>
            <td>{display}</td>
            {metric_cell(m['accuracy'])}
            {metric_cell(m.get('macro_precision', 0))}
            {metric_cell(m.get('macro_recall', 0))}
            {metric_cell(m.get('macro_f1', 0))}
            {metric_cell(m.get('weighted_f1', 0))}
            <td>{m.get('num_evaluated', 0):,}</td>
        </tr>"""

    # Per-class detail sections
    class_sections = ""
    for task_key, display in TASK_DISPLAY.items():
        m = bert.get(task_key, {})
        if not m:
            continue
        report = m.get("classification_report", {})
        classes = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
        if not classes:
            continue
        rows = ""
        for cls in classes:
            r = report[cls]
            rows += f"""<tr>
                <td>{cls}</td>
                <td>{r['precision']:.4f}</td>
                <td>{r['recall']:.4f}</td>
                <td>{r['f1-score']:.4f}</td>
                <td>{int(r['support']):,}</td>
            </tr>"""
        class_sections += f"""
        <h3 style="margin-top:1.5rem;color:#1e293b">{display}</h3>
        <table><thead><tr>
            <th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>
        </tr></thead><tbody>{rows}</tbody></table>"""

    fig_rel = fig_dir.name

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evaluation Report — Multi-Task Sentiment Analysis</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,-apple-system,sans-serif; background:#f8fafc; color:#334155; line-height:1.6; padding:2rem 3rem; max-width:1200px; margin:auto; }}
  h1 {{ font-size:1.8rem; color:#0f172a; border-bottom:3px solid #2563eb; padding-bottom:0.5rem; margin-bottom:1.5rem; }}
  h2 {{ font-size:1.3rem; color:#1e293b; margin:2rem 0 0.8rem; padding-left:0.5rem; border-left:4px solid #2563eb; }}
  h3 {{ font-size:1.05rem; }}
  table {{ border-collapse:collapse; width:100%; margin:0.5rem 0 1.5rem; font-size:0.9rem; table-layout:fixed; }}
  th {{ background:#1e293b; color:white; padding:0.6rem 1rem; text-align:right; font-weight:600; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:0.5rem 1rem; border-bottom:1px solid #e2e8f0; text-align:right; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f1f5f9; }}
  .highlight {{ background:#eff6ff !important; font-weight:700; }}
  .avg-row td {{ border-top:2px solid #1e293b; font-weight:700; }}
  .chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin:1rem 0; }}
  .chart-grid img {{ width:100%; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
  .chart-full img {{ width:100%; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); margin:1rem 0; }}
  .badge {{ display:inline-block; background:#2563eb; color:white; padding:0.15rem 0.6rem; border-radius:12px; font-size:0.8rem; font-weight:600; margin-left:0.5rem; }}
  .metric-cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1rem 0 2rem; }}
  .metric-card {{ background:white; border-radius:10px; padding:1.2rem; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06); border-top:3px solid #2563eb; }}
  .metric-card .value {{ font-size:1.8rem; font-weight:800; color:#0f172a; }}
  .metric-card .label {{ font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; }}
  .metric-card .task {{ font-size:0.85rem; color:#2563eb; font-weight:600; margin-bottom:0.2rem; }}
</style>
</head>
<body>

<h1>Evaluation Report</h1>

<div class="metric-cards">
  <div class="metric-card">
    <div class="task">Sentiment</div>
    <div class="value">{bert.get('sentiment', {}).get('macro_f1', 0):.1%}</div>
    <div class="label">Macro F1</div>
  </div>
  <div class="metric-card">
    <div class="task">Emotion</div>
    <div class="value">{bert.get('emotion', {}).get('macro_f1', 0):.1%}</div>
    <div class="label">Macro F1</div>
  </div>
  <div class="metric-card">
    <div class="task">Intensity</div>
    <div class="value">{bert.get('intensity', {}).get('macro_f1', 0):.1%}</div>
    <div class="label">Macro F1</div>
  </div>
  <div class="metric-card">
    <div class="task">Topic</div>
    <div class="value">{bert.get('topic', {}).get('macro_f1', 0):.1%}</div>
    <div class="label">Macro F1</div>
  </div>
</div>

<h2>1. Model Comparison &mdash; Sentiment Polarity</h2>
<table>
<thead><tr>
  <th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Weighted F1</th><th>Samples</th>
</tr></thead>
<tbody>{sent_rows}</tbody>
</table>

<div class="chart-full"><img src="{fig_rel}/model_comparison.png" alt="Model Comparison"></div>

<h2>2. Multi-Head BERT &mdash; Per-Task Breakdown</h2>
<table>
<thead><tr>
  <th>Task</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Macro F1</th><th>Weighted F1</th><th>Samples</th>
</tr></thead>
<tbody>
  {task_rows}
</tbody>
</table>

<div class="chart-full"><img src="{fig_rel}/bert_per_task.png" alt="BERT Per-Task"></div>

<h2>3. Per-Class F1 Scores</h2>
{class_sections}

<div class="chart-full"><img src="{fig_rel}/per_class_f1.png" alt="Per-Class F1"></div>

<h2>4. Confusion Matrices</h2>
<div class="chart-full"><img src="{fig_rel}/confusion_matrices.png" alt="Confusion Matrices"></div>

</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────

@click.command()
@click.option("--input", "input_path", default="evaluation_results.json")
@click.option("--output-dir", default="evaluation_figures")
@click.option("--report", default="evaluation_report.html")
def main(input_path: str, output_dir: str, report: str):
    data = load_results(input_path)
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print_summary_table(data)

    if not HAS_MPL:
        print("[WARN] matplotlib not installed — skipping chart generation.")
        print("       Install with: pip install matplotlib seaborn")
        return

    print("Generating charts...")
    plot_model_comparison(data, fig_dir)
    plot_bert_tasks(data, fig_dir)
    plot_confusion_matrices(data, fig_dir)
    plot_per_class_f1(data, fig_dir)
    print(f"  Charts saved to {fig_dir}/")

    report_path = Path(report).resolve()
    generate_html_report(data, fig_dir, report_path)
    print(f"  HTML report saved to {report}")

    webbrowser.open(report_path.as_uri())
    print(f"  Report opened in browser.\n")


if __name__ == "__main__":
    main()
