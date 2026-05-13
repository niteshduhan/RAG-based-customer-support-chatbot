"""
report_visuals.py — Presentation-grade evaluation visualisations
for the Amazon Customer Service RAG Agent.

Metrics actually produced by eval.py:
    avg_answer_similarity   — cosine(answer, ground_truth)   via e5-base
    avg_context_relevance   — cosine(query, each chunk)       via e5-base
    avg_latency             — wall-clock seconds per query
    category_scores         — dict[category → avg similarity]
    per_query               — list of per-question scores

Bugs fixed vs original:
  ✗ avg_faithfulness        → key did NOT exist in eval.py output  (removed)
  ✗ avg_no_hallucination    → key did NOT exist in eval.py output  (removed)
  ✓ Replaced with avg_context_relevance throughout
  ✓ Radar uses 4 real axes: answer_sim, ctx_relevance, speed, hindi
  ✓ All dynamic y-axes computed from actual data
  ✓ Dark presentation theme matching project aesthetic
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

os.makedirs("report_visuals", exist_ok=True)

# ── Load results ───────────────────────────────────────────────
with open("eval_results.json", "r") as f:
    data = json.load(f)

# ── Palette & labels ────────────────────────────────────────────
COLORS  = ["#1abc9c", "#7c5cbf", "#e67e22"]          # green / purple / orange
LABELS  = ["LLaMA 4 Scout 17B\n(primary)",
           "LLaMA 3.3 70B\n(comparison A)",
           "LLaMA 3.1 8B\n(comparison B)"]
SHORT   = ["Scout 17B", "LLaMA 70B", "LLaMA 8B"]

CATS      = ["basic_policy", "multi_hop", "edge_case", "prime",
             "promotions", "delivery", "hindi", "account"]
CAT_LABELS = ["Basic\nPolicy", "Multi-Hop\nReasoning", "Edge\nCases",
              "Prime\nFAQ", "Promo-\ntions", "Delivery\n& Tracking",
              "Hindi\n(multilingual)", "Account\nPolicy"]

# ── Dark theme ─────────────────────────────────────────────────
BG     = "#0b0f1a"
PANEL  = "#0f1623"
GRID   = "#1e2535"
TEXT   = "#e8eaf6"
SUBDIM = "#8892b0"

plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     PANEL,
    "axes.edgecolor":     GRID,
    "axes.labelcolor":    TEXT,
    "xtick.color":        SUBDIM,
    "ytick.color":        SUBDIM,
    "text.color":         TEXT,
    "grid.color":         GRID,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
    "grid.alpha":         1.0,
    "axes.grid":          True,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "figure.dpi":         150,
    "savefig.facecolor":  BG,
    "savefig.edgecolor":  BG,
    "legend.facecolor":   PANEL,
    "legend.edgecolor":   GRID,
    "legend.labelcolor":  TEXT,
})


def dynamic_ylim(values, pad_lo=0.04, pad_hi=0.06):
    lo = min(values) - pad_lo
    hi = max(values) + pad_hi
    return max(0.0, round(lo - lo % 0.01, 2)), min(1.02, round(hi, 2))


def bar_label(ax, bars, fmt="{:.4f}", yoff=0.003, fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + yoff, fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold",
                color=TEXT)


def subtitle(ax, text):
    ax.set_title(text, fontsize=13, fontweight="bold",
                 color=TEXT, pad=12, loc="left")


# ══════════════════════════════════════════════════════════════
# 01 — Overall Metrics: answer_similarity + context_relevance
# ══════════════════════════════════════════════════════════════
def plot_overall_metrics():
    metric_keys   = ["avg_answer_similarity", "avg_context_relevance"]
    metric_labels = ["Answer Similarity\n(vs ground truth)", "Context Relevance\n(query↔chunk cosine)"]

    vals_per_model = [[d[k] for k in metric_keys] for d in data]
    flat           = [v for row in vals_per_model for v in row]
    ylo, yhi       = dynamic_ylim(flat)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("RAG Evaluation — Overall Metrics Comparison",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.98)

    x = np.arange(len(metric_labels))
    w = 0.24
    for i, (vals, color, label) in enumerate(zip(vals_per_model, COLORS, LABELS)):
        bars = ax.bar(x + i * w, vals, w,
                      color=color, alpha=0.88,
                      label=label.replace("\n", " "),
                      zorder=3, linewidth=0)
        bar_label(ax, bars)

    ax.set_xticks(x + w)
    ax.set_xticklabels(metric_labels, fontsize=12, color=TEXT)
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("Score  (cosine similarity, 0–1)", color=SUBDIM, fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.6)
    ax.set_axisbelow(True)

    # annotation: winner call-out
    best_sim = max(d["avg_answer_similarity"] for d in data)
    ax.axhline(best_sim, color=COLORS[0], linestyle=":", linewidth=1.2, alpha=0.5)
    ax.text(len(metric_labels) - 0.05, best_sim + 0.002,
            f"Best sim: {best_sim:.4f}", color=COLORS[0],
            fontsize=8, ha="right", alpha=0.9)

    plt.tight_layout()
    plt.savefig("report_visuals/01_overall_metrics.png", bbox_inches="tight")
    plt.close()
    print("✅  01_overall_metrics.png")


# ══════════════════════════════════════════════════════════════
# 02 — Category Breakdown (answer similarity per category)
# ══════════════════════════════════════════════════════════════
def plot_category_breakdown():
    cat_vals = [[d["category_scores"][c] for c in CATS] for d in data]
    flat     = [v for row in cat_vals for v in row]
    ylo, yhi = dynamic_ylim(flat, pad_lo=0.03, pad_hi=0.07)

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle("Answer Similarity by Question Category",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.98)

    x = np.arange(len(CATS))
    w = 0.24
    for i, (vals, color, label) in enumerate(zip(cat_vals, COLORS, LABELS)):
        bars = ax.bar(x + i * w, vals, w,
                      color=color, alpha=0.88,
                      label=label.replace("\n", " "),
                      zorder=3, linewidth=0)
        bar_label(ax, bars, fmt="{:.3f}", fontsize=7.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels(CAT_LABELS, fontsize=10, color=TEXT)
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("Answer Similarity Score", color=SUBDIM, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.6, loc="lower right")
    ax.set_axisbelow(True)

    # shade hardest category
    worst_cat = min(range(len(CATS)),
                    key=lambda i: np.mean([d["category_scores"][CATS[i]] for d in data]))
    ax.axvspan(worst_cat - 0.15, worst_cat + 0.75,
               color="#e74c3c", alpha=0.07, zorder=0)
    ax.text(worst_cat + 0.3, yhi - 0.003,
            "Hardest\ncategory", ha="center", va="top",
            fontsize=8, color="#e74c3c", alpha=0.8)

    plt.tight_layout()
    plt.savefig("report_visuals/02_category_breakdown.png", bbox_inches="tight")
    plt.close()
    print("✅  02_category_breakdown.png")


# ══════════════════════════════════════════════════════════════
# 03 — Quality vs Speed Scatter
# ══════════════════════════════════════════════════════════════
def plot_latency_vs_quality():
    lats = [d["avg_latency"]           for d in data]
    sims = [d["avg_answer_similarity"] for d in data]
    ctxr = [d["avg_context_relevance"] for d in data]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("Quality vs Speed Trade-off",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.98)

    ylo, yhi = dynamic_ylim(sims, pad_lo=0.02, pad_hi=0.02)
    xlo = min(lats) - 0.12
    xhi = max(lats) + 0.22

    # bubble size ~ context relevance
    for d, color, label, short in zip(data, COLORS, LABELS, SHORT):
        size = (d["avg_context_relevance"] * 800) ** 1.2
        ax.scatter(d["avg_latency"], d["avg_answer_similarity"],
                   s=size, color=color, zorder=5,
                   alpha=0.85, edgecolors="white", linewidth=1.2)
        ax.annotate(f"  {short}\nctx_rel={d['avg_context_relevance']:.3f}",
                    (d["avg_latency"], d["avg_answer_similarity"]),
                    textcoords="offset points", xytext=(14, -4),
                    fontsize=9, color=color, fontweight="bold")

    # quadrant lines
    mean_lat = np.mean(lats)
    mean_sim = np.mean(sims)
    ax.axhline(mean_sim, color=GRID, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axvline(mean_lat, color=GRID, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.text(xlo + 0.02, yhi - 0.002, "High Quality\nFast ✓",
            fontsize=8, color="#1abc9c", alpha=0.85, va="top")
    ax.text(xhi - 0.02, yhi - 0.002, "High Quality\nSlow",
            fontsize=8, color="#e67e22", alpha=0.85, va="top", ha="right")

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Avg Latency (seconds) — lower is better", color=SUBDIM, fontsize=11)
    ax.set_ylabel("Avg Answer Similarity — higher is better", color=SUBDIM, fontsize=11)

    # bubble size legend
    ax.text(0.02, 0.04,
            "Bubble size ∝ Context Relevance score",
            transform=ax.transAxes, fontsize=8,
            color=SUBDIM, alpha=0.8)

    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("report_visuals/03_latency_vs_quality.png", bbox_inches="tight")
    plt.close()
    print("✅  03_latency_vs_quality.png")


# ══════════════════════════════════════════════════════════════
# 04 — Per-query Heatmap (answer similarity × model)
# ══════════════════════════════════════════════════════════════
def plot_per_query_heatmap():
    # short question labels (≤22 chars)
    q_labels = [
        "Phone cracked?", "Wrong color shoes?", "Items for return?",
        "Intl defective laptop?", "No stock repl.?", "Noisy fridge?",
        "TV sale 3 wks?", "Used laptop defect?", "Non-ret damaged?",
        "OTP shared?", "Prime benefits?", "Cancel Prime?",
        "Prime fee?", "Charged w/o signup?", "Coupon refund?",
        "Cashback on return?", "Multi promo code?", "Shows delivered?",
        "Track order?", "10d in transit?", "Damaged (Hindi)",
        "Prime cancel (Hindi)", "Refund time (Hindi)", "10+ returns?",
        "Gift return?"
    ]

    matrix = np.array([
        [r["answer_similarity"] for r in d["per_query"]]
        for d in data
    ])

    vmin = max(0.75, matrix.min() - 0.01)
    vmax = min(1.00, matrix.max() + 0.005)

    # custom green-to-red-ish colormap on dark background
    cmap = LinearSegmentedColormap.from_list(
        "rag_heat",
        ["#7b241c", "#e67e22", "#f1c40f", "#27ae60", "#1abc9c"]
    )

    fig, ax = plt.subplots(figsize=(18, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    fig.suptitle("Per-Query Answer Similarity Heatmap (all 25 questions × 3 models)",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.01)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_yticks(range(3))
    ax.set_yticklabels(SHORT, fontsize=10, color=TEXT)
    ax.set_xticks(range(len(q_labels)))
    ax.set_xticklabels(q_labels, rotation=42, ha="right", fontsize=8, color=TEXT)

    # cell labels
    for i in range(3):
        for j in range(len(q_labels)):
            val = matrix[i, j]
            txt_col = "black" if val > (vmin + vmax) * 0.55 else "white"
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=7, color=txt_col, fontweight="bold")

    # category separator lines (3 per category: basic=0-2, multi=3-5, ...)
    cat_ends = [2, 5, 9, 13, 16, 19, 22, 24]
    for sep in cat_ends[:-1]:
        ax.axvline(sep + 0.5, color=BG, linewidth=2)

    cb = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.015)
    cb.set_label("Answer Similarity", color=SUBDIM, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=SUBDIM)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=8)

    plt.tight_layout()
    plt.savefig("report_visuals/04_per_query_heatmap.png", bbox_inches="tight")
    plt.close()
    print("✅  04_per_query_heatmap.png")


# ══════════════════════════════════════════════════════════════
# 05 — Radar Chart (4 real axes from eval.py output)
# ══════════════════════════════════════════════════════════════
def plot_radar():
    """
    Axes (all from eval.py keys):
        Answer Similarity   — avg_answer_similarity
        Context Relevance   — avg_context_relevance
        Speed (norm.)       — 1 - latency/max_latency
        Hindi Score         — category_scores["hindi"]
    """
    axis_labels = ["Answer\nSimilarity", "Context\nRelevance",
                   "Speed\n(norm.)", "Hindi\nScore"]
    n = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    max_lat = max(d["avg_latency"] for d in data)

    def get_vals(d):
        speed = 1.0 - (d["avg_latency"] / max_lat)
        return [
            d["avg_answer_similarity"],
            d["avg_context_relevance"],
            speed,
            d["category_scores"]["hindi"],
        ]

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True),
                           facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # grid and spoke styling
    ax.set_rlabel_position(30)
    all_v = [v for d in data for v in get_vals(d)]
    y_max = min(1.0, max(all_v) + 0.05)
    y_min = max(0.0, min(all_v) - 0.05)
    ticks = np.round(np.linspace(y_min, y_max, 4), 2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.2f}" for t in ticks],
                       fontsize=7, color=SUBDIM)
    ax.set_ylim(y_min - 0.02, y_max + 0.02)

    ax.set_thetagrids(np.degrees(angles[:-1]), axis_labels,
                      fontsize=11, color=TEXT)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.9)

    for d, color, label in zip(data, COLORS, LABELS):
        vals  = get_vals(d)
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2.2,
                label=label.replace("\n", " "))
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_title("Model Capability Radar\n(4 metrics from eval.py)",
                 fontsize=13, fontweight="bold", color=TEXT, pad=22)
    ax.legend(loc="upper right",
              bbox_to_anchor=(1.38, 1.15),
              fontsize=9, framealpha=0.5)

    plt.tight_layout()
    plt.savefig("report_visuals/05_radar_chart.png", bbox_inches="tight")
    plt.close()
    print("✅  05_radar_chart.png")


# ══════════════════════════════════════════════════════════════
# 06 — Latency Distribution (violin / strip plot per model)
# ══════════════════════════════════════════════════════════════
def plot_latency_distribution():
    """
    Per-query latency distribution for each model.
    Shows mean + spread — important for production SLA decisions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle("Per-Query Latency Distribution by Model",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.98)

    for ax, d, color, short in zip(axes, data, COLORS, SHORT):
        lats = [r["latency"] for r in d["per_query"]]
        mean = np.mean(lats)
        std  = np.std(lats)

        # Histogram
        ax.hist(lats, bins=8, color=color, alpha=0.75,
                edgecolor=BG, linewidth=0.8, zorder=3)
        ax.axvline(mean, color="white", linestyle="--",
                   linewidth=1.6, alpha=0.85, zorder=5)
        ax.text(mean + 0.005, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1,
                f"μ={mean:.3f}s\nσ={std:.3f}s",
                color="white", fontsize=8.5, va="top", ha="left")

        ax.set_title(short, fontsize=12, fontweight="bold", color=color, pad=8)
        ax.set_xlabel("Latency (s)", color=SUBDIM, fontsize=10)
        ax.set_ylabel("# Queries", color=SUBDIM, fontsize=10)
        ax.tick_params(colors=SUBDIM)
        ax.set_facecolor(PANEL)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("report_visuals/06_latency_distribution.png", bbox_inches="tight")
    plt.close()
    print("✅  06_latency_distribution.png")


# ══════════════════════════════════════════════════════════════
# 07 — Summary Dashboard (2×3 grid, all KPIs at a glance)
# ══════════════════════════════════════════════════════════════
def plot_dashboard():
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle("Amazon RAG Customer Service Agent — Evaluation Report",
                 fontsize=17, fontweight="bold", color=TEXT, y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

    # ── top-left: answer similarity bar ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sims = [d["avg_answer_similarity"] for d in data]
    ylo, yhi = dynamic_ylim(sims, pad_lo=0.03, pad_hi=0.04)
    bars = ax1.bar(range(3), sims, color=COLORS, alpha=0.88, linewidth=0)
    bar_label(ax1, bars, fontsize=8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(SHORT, fontsize=9, color=TEXT)
    ax1.set_ylim(ylo, yhi)
    ax1.set_title("Answer Similarity", fontsize=12, fontweight="bold",
                  color=TEXT, pad=8, loc="left")
    ax1.set_ylabel("Score", color=SUBDIM, fontsize=9)
    ax1.set_axisbelow(True)

    # ── top-middle: context relevance bar ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ctxr = [d["avg_context_relevance"] for d in data]
    ylo2, yhi2 = dynamic_ylim(ctxr, pad_lo=0.03, pad_hi=0.04)
    bars2 = ax2.bar(range(3), ctxr, color=COLORS, alpha=0.88, linewidth=0)
    bar_label(ax2, bars2, fontsize=8)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(SHORT, fontsize=9, color=TEXT)
    ax2.set_ylim(ylo2, yhi2)
    ax2.set_title("Context Relevance", fontsize=12, fontweight="bold",
                  color=TEXT, pad=8, loc="left")
    ax2.set_ylabel("Score", color=SUBDIM, fontsize=9)
    ax2.set_axisbelow(True)

    # ── top-right: latency bar ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    lats = [d["avg_latency"] for d in data]
    bars3 = ax3.bar(range(3), lats, color=COLORS, alpha=0.88, linewidth=0)
    bar_label(ax3, bars3, fmt="{:.3f}s", fontsize=8)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(SHORT, fontsize=9, color=TEXT)
    ax3.set_ylim(0, max(lats) + 0.25)
    ax3.set_title("Avg Response Latency", fontsize=12, fontweight="bold",
                  color=TEXT, pad=8, loc="left")
    ax3.set_ylabel("Seconds", color=SUBDIM, fontsize=9)
    ax3.set_axisbelow(True)

    # ── bottom: category breakdown (spans all 3 cols) ────────
    ax4 = fig.add_subplot(gs[1, :])
    cat_vals = [[d["category_scores"][c] for c in CATS] for d in data]
    flat4    = [v for row in cat_vals for v in row]
    ylo4, yhi4 = dynamic_ylim(flat4, pad_lo=0.02, pad_hi=0.06)
    x = np.arange(len(CATS))
    w = 0.25
    for i, (vals, color, label) in enumerate(zip(cat_vals, COLORS, LABELS)):
        bars4 = ax4.bar(x + i * w, vals, w,
                        color=color, alpha=0.88,
                        label=label.replace("\n", " "), linewidth=0)
        bar_label(ax4, bars4, fmt="{:.3f}", fontsize=7.5, yoff=0.001)

    ax4.set_xticks(x + w)
    ax4.set_xticklabels(CAT_LABELS, fontsize=10, color=TEXT)
    ax4.set_ylim(ylo4, yhi4)
    ax4.set_ylabel("Answer Similarity", color=SUBDIM, fontsize=10)
    ax4.set_title("Category Breakdown — Answer Similarity per Evaluation Category",
                  fontsize=12, fontweight="bold", color=TEXT, pad=8, loc="left")
    ax4.legend(fontsize=9, framealpha=0.5, loc="lower right")
    ax4.set_axisbelow(True)

    # ── model colour legend at bottom ────────────────────────
    patches = [mpatches.Patch(color=c, label=l.replace("\n", " "))
               for c, l in zip(COLORS, LABELS)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.01),
               framealpha=0.4, facecolor=PANEL, edgecolor=GRID)

    plt.savefig("report_visuals/07_dashboard.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("✅  07_dashboard.png")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n Generating presentation-grade report visuals...\n")
    plot_overall_metrics()
    plot_category_breakdown()
    plot_latency_vs_quality()
    plot_per_query_heatmap()
    plot_radar()
    plot_latency_distribution()
    plot_dashboard()

    print("\n" + "═" * 60)
    print("  ALL 7 VISUALS SAVED → report_visuals/")
    print("═" * 60)
    print("  01_overall_metrics.png      — answer_sim + ctx_relevance bars")
    print("  02_category_breakdown.png   — 8-category grouped bar")
    print("  03_latency_vs_quality.png   — bubble scatter, quadrant lines")
    print("  04_per_query_heatmap.png    — 25q × 3 model heatmap")
    print("  05_radar_chart.png          — 4-axis radar (real metrics)")
    print("  06_latency_distribution.png — per-model latency histogram")
    print("  07_dashboard.png            — full KPI dashboard")