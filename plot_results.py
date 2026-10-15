"""Draw reproducible scientific figures from saved measurements."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    cache = json.loads((ROOT / "results/cache_benchmark/summary.json").read_text())
    generation = json.loads((ROOT / "results/generation_summary.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {"disabled": "#65768b", "reset_each_request": "#e39a34", "persistent": "#167f79"}
    labels = {"disabled": "Cache disabled", "reset_each_request": "Cache reset each request",
              "persistent": "Cache retained across requests"}
    for ax, mode in zip(axes, ["static", "dynamic"]):
        for policy in colors:
            rows = sorted([x for x in cache["condition_summaries"] if x["mode"] == mode
                           and x["cache_policy"] == policy and x["phase"] == "after_first"],
                          key=lambda x: x["tool_count"])
            ax.plot([x["tool_count"] for x in rows], [x["median_compile_ms"] for x in rows],
                    "o-", color=colors[policy], label=labels[policy])
        ax.set(title=f"{mode.title()} tool sets", xlabel="Tools per request", ylabel="Median compilation (ms)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("XGrammar-2 on Apple M3 Pro: subsequent requests, 3 workload repeats", fontsize=12)
    fig.savefig(ROOT / "results/cache_comparison.png", dpi=180)
    plt.close(fig)
    modes = ["unconstrained", "xgrammar"]
    metrics = ["schema_valid", "argument_exact", "content_argument_exact"]
    metric_labels = ["Strict schema\ncompliance", "Strict tool-call\nsuccess", "Argument match\n(fence relaxed)"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    x = np.arange(3)
    for offset, mode, color, label in [(-.19, modes[0], "#65768b", "Unconstrained"),
                                       (.19, modes[1], "#167f79", "XGrammar constrained")]:
        scores = generation["metrics"][mode]
        values = [scores[k] / scores["n"] * 100 for k in metrics]
        bars = ax.bar(x + offset, values, .36, color=color, label=label)
        for bar, key in zip(bars, metrics):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                    f"{scores[key]}/{scores['n']}", ha="center", fontsize=10)
    ax.set(xticks=x, xticklabels=metric_labels, ylim=(0, 112), ylabel="Cases (%)",
           title="Qwen3-0.6B: fixed first 32 BFCL-v3 simple cases")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.13), ncol=2, frameon=False)
    fig.savefig(ROOT / "results/generation_comparison.png", dpi=180)
    plt.close(fig)
    print("Saved cache_comparison.png and generation_comparison.png")


if __name__ == "__main__":
    main()
