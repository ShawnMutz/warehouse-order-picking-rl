"""
Create final dissertation figures from the frozen analysis outputs.

Reads:
    results/final/final_evaluation/analysis/
        completion_summary.csv
        paired_route_quality_summary.csv
        paired_heuristic_tests.csv

Writes:
    results/final/final_evaluation/figures/

Figures:
    dqn_completion_by_order_size
    paired_optimality_gap_uniform
    paired_optimality_gap_clustered
    dqn_paired_distance_difference
    dqn_win_tie_loss

Each figure is saved as PNG (300 dpi) and PDF.
No training, simulation, or new statistical testing is performed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "final"
    / "final_evaluation"
)

ANALYSIS_DIR = FINAL_DIR / "analysis"
FIGURES_DIR = FINAL_DIR / "figures"

COMPLETION_PATH = ANALYSIS_DIR / "completion_summary.csv"
ROUTE_QUALITY_PATH = ANALYSIS_DIR / "paired_route_quality_summary.csv"
PAIRED_TESTS_PATH = ANALYSIS_DIR / "paired_heuristic_tests.csv"

ORDER_SIZES = [5, 10, 15, 20]
DISTRIBUTIONS = ["uniform", "clustered"]
METHODS = ["DQN", "S-shape", "Return", "Largest Gap"]
HEURISTICS = ["S-shape", "Return", "Largest Gap"]
PNG_DPI = 300


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n"
            "Run experiments.analyze_final_results first."
        )


def load_inputs():
    for path in [
        COMPLETION_PATH,
        ROUTE_QUALITY_PATH,
        PAIRED_TESTS_PATH,
    ]:
        require_file(path)

    completion = pd.read_csv(COMPLETION_PATH)
    route_quality = pd.read_csv(ROUTE_QUALITY_PATH)
    paired_tests = pd.read_csv(PAIRED_TESTS_PATH)

    return completion, route_quality, paired_tests


def save_figure(fig, stem):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=PNG_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
    )
    plt.close(fig)

    return png, pdf


def create_completion_figure(completion):
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111)

    for distribution in DISTRIBUTIONS:
        sub = (
            completion[
                completion["distribution"] == distribution
            ]
            .sort_values("order_size")
        )

        x = sub["order_size"].to_numpy(dtype=float)
        y = sub["completion_rate"].to_numpy(dtype=float)

        lower = sub["wilson_95_lower"].to_numpy(dtype=float)
        upper = sub["wilson_95_upper"].to_numpy(dtype=float)

        yerr = np.vstack([
            y - lower,
            upper - y,
        ])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=distribution.capitalize(),
        )

    ax.set_title("DQN Completion Rate by Order Size")
    ax.set_xlabel("Order size (number of picks)")
    ax.set_ylabel("Completion rate (%)")
    ax.set_xticks(ORDER_SIZES)
    ax.set_ylim(0, 105)
    ax.legend(title="Distribution")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    return save_figure(
        fig,
        "dqn_completion_by_order_size",
    )


def create_optimality_gap_figure(
    route_quality,
    distribution,
):
    sub = (
        route_quality[
            route_quality["distribution"] == distribution
        ]
        .sort_values("order_size")
    )

    x = np.arange(len(ORDER_SIZES))
    width = 0.19

    col = {
        "DQN": "dqn_mean_gap",
        "S-shape": "s_shape_mean_gap",
        "Return": "return_mean_gap",
        "Largest Gap": "largest_gap_mean_gap",
    }

    offsets = (
        np.arange(len(METHODS))
        - (len(METHODS) - 1) / 2
    ) * width

    fig = plt.figure(figsize=(9.5, 5.8))
    ax = fig.add_subplot(111)

    for offset, method in zip(offsets, METHODS):
        ax.bar(
            x + offset,
            sub[col[method]].to_numpy(dtype=float),
            width,
            label=method,
        )

    ax.set_title(
        f"Mean Optimality Gap — {distribution.capitalize()} Orders"
    )
    ax.set_xlabel("Order size (number of picks)")
    ax.set_ylabel("Mean optimality gap (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER_SIZES)
    ax.legend(title="Method")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    return save_figure(
        fig,
        f"paired_optimality_gap_{distribution}",
    )


def create_paired_distance_difference_figure(paired_tests):
    conditions = [
        ("uniform", 5, "U5"),
        ("uniform", 10, "U10"),
        ("uniform", 15, "U15"),
        ("uniform", 20, "U20"),
        ("clustered", 5, "C5"),
        ("clustered", 10, "C10"),
        ("clustered", 15, "C15"),
        ("clustered", 20, "C20"),
    ]

    x = np.arange(len(conditions))
    width = 0.24

    offsets = (
        np.arange(len(HEURISTICS))
        - (len(HEURISTICS) - 1) / 2
    ) * width

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    for offset, heuristic in zip(offsets, HEURISTICS):
        values = []

        for distribution, size, _ in conditions:
            row = paired_tests[
                (paired_tests["distribution"] == distribution)
                & (paired_tests["order_size"] == size)
                & (paired_tests["heuristic"] == heuristic)
            ]

            if len(row) != 1:
                raise RuntimeError(
                    f"Expected one row for "
                    f"{distribution}, {size}, {heuristic}."
                )

            values.append(
                float(
                    row.iloc[0][
                        "mean_dqn_minus_heuristic_distance"
                    ]
                )
            )

        ax.bar(
            x + offset,
            values,
            width,
            label=heuristic,
        )

    ax.axhline(0, linewidth=1)
    ax.set_title(
        "Mean Paired Distance Difference: DQN Minus Heuristic"
    )
    ax.set_xlabel("Final-test condition")
    ax.set_ylabel("Mean distance difference")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label for _, _, label in conditions]
    )
    ax.legend(title="Heuristic")
    ax.grid(axis="y", alpha=0.25)

    ax.text(
        0.01,
        0.98,
        "Positive = DQN longer; negative = DQN shorter",
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()

    return save_figure(
        fig,
        "dqn_paired_distance_difference",
    )


def create_win_tie_loss_figure(paired_tests):
    agg = (
        paired_tests.groupby(
            "heuristic",
            as_index=False,
        )[
            [
                "dqn_wins",
                "ties",
                "dqn_losses",
            ]
        ]
        .sum()
        .set_index("heuristic")
        .reindex(HEURISTICS)
        .reset_index()
    )

    x = np.arange(len(HEURISTICS))

    wins = agg["dqn_wins"].to_numpy(dtype=float)
    ties = agg["ties"].to_numpy(dtype=float)
    losses = agg["dqn_losses"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(8.5, 5.8))
    ax = fig.add_subplot(111)

    ax.bar(
        x,
        wins,
        label="DQN wins",
    )
    ax.bar(
        x,
        ties,
        bottom=wins,
        label="Ties",
    )
    ax.bar(
        x,
        losses,
        bottom=wins + ties,
        label="DQN losses",
    )

    ax.set_title(
        "DQN Paired Outcomes Across Final-Test Conditions"
    )
    ax.set_xlabel("Comparator heuristic")
    ax.set_ylabel(
        "Number of paired completed-route comparisons"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(HEURISTICS)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    return save_figure(
        fig,
        "dqn_win_tie_loss",
    )


def save_manifest(records):
    path = FIGURES_DIR / "figure_manifest.csv"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "figure",
                "title",
                "png",
                "pdf",
                "source",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    return path


def save_config():
    path = FIGURES_DIR / "figure_config.json"

    config = {
        "png_dpi": PNG_DPI,
        "order_sizes": ORDER_SIZES,
        "distributions": DISTRIBUTIONS,
        "methods": METHODS,
        "completion_error_bars":
            "Wilson 95% confidence intervals",
        "optimality_gap_subset":
            "completed DQN orders only",
        "paired_distance_definition":
            "DQN distance - heuristic distance",
        "additional_training": False,
        "additional_simulation": False,
        "additional_statistical_testing": False,
    }

    with path.open("w") as f:
        json.dump(
            config,
            f,
            indent=4,
        )

    return path


def main():
    print(
        "\n"
        "================================================\n"
        "CREATE FINAL DISSERTATION FIGURES\n"
        "================================================"
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    completion, route_quality, paired_tests = (
        load_inputs()
    )

    records = []

    png, pdf = create_completion_figure(
        completion
    )
    records.append(
        {
            "figure": 1,
            "title":
                "DQN Completion Rate by Order Size",
            "png": str(png),
            "pdf": str(pdf),
            "source": str(COMPLETION_PATH),
            "note":
                "Uniform vs clustered with Wilson 95% CI.",
        }
    )

    png, pdf = create_optimality_gap_figure(
        route_quality,
        "uniform",
    )
    records.append(
        {
            "figure": 2,
            "title":
                "Mean Optimality Gap — Uniform Orders",
            "png": str(png),
            "pdf": str(pdf),
            "source": str(ROUTE_QUALITY_PATH),
            "note":
                "All methods use the same completed-DQN orders.",
        }
    )

    png, pdf = create_optimality_gap_figure(
        route_quality,
        "clustered",
    )
    records.append(
        {
            "figure": 3,
            "title":
                "Mean Optimality Gap — Clustered Orders",
            "png": str(png),
            "pdf": str(pdf),
            "source": str(ROUTE_QUALITY_PATH),
            "note":
                "All methods use the same completed-DQN orders.",
        }
    )

    png, pdf = create_paired_distance_difference_figure(
        paired_tests
    )
    records.append(
        {
            "figure": 4,
            "title":
                "Mean Paired Distance Difference",
            "png": str(png),
            "pdf": str(pdf),
            "source": str(PAIRED_TESTS_PATH),
            "note":
                "Positive values mean DQN is longer.",
        }
    )

    png, pdf = create_win_tie_loss_figure(
        paired_tests
    )
    records.append(
        {
            "figure": 5,
            "title":
                "DQN Paired Win/Tie/Loss Outcomes",
            "png": str(png),
            "pdf": str(pdf),
            "source": str(PAIRED_TESTS_PATH),
            "note":
                "Completed DQN routes only.",
        }
    )

    manifest = save_manifest(records)
    config = save_config()

    print(f"Created {len(records)} figures.")

    for record in records:
        print(
            f"\nFigure {record['figure']}: "
            f"{record['title']}"
        )
        print(f"  PNG: {record['png']}")
        print(f"  PDF: {record['pdf']}")

    print(f"\nManifest: {manifest}")
    print(f"Config:   {config}")

    print(
        "\n"
        "================================================\n"
        "FINAL FIGURES COMPLETE\n"
        "================================================"
    )


if __name__ == "__main__":
    main()
