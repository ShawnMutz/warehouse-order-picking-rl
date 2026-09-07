"""
Post-final-test statistical analysis.

Reads:
    results/final/final_evaluation/final_results.csv

Performs no training or simulation.

Outputs:
    results/final/final_evaluation/analysis/
        completion_summary.csv
        overall_completion_summary.csv
        completion_distribution_tests.csv
        paired_heuristic_tests.csv
        paired_route_quality_summary.csv
        analysis_summary.txt
        analysis_config.json
"""

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact, wilcoxon
except ImportError as exc:
    raise ImportError(
        "SciPy is required. Install with:\n"
        "    python -m pip install scipy"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "final"
    / "final_evaluation"
)

INPUT_PATH = FINAL_DIR / "final_results.csv"
ANALYSIS_DIR = FINAL_DIR / "analysis"

ORDER_SIZES = [5, 10, 15, 20]
DISTRIBUTIONS = ["uniform", "clustered"]
HEURISTICS = ["S-shape", "Return", "Largest Gap"]

ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95
TOL = 1e-9

DISTANCE_COL = {
    "S-shape": "s_shape_distance",
    "Return": "return_distance",
    "Largest Gap": "largest_gap_distance",
}

IMPROVEMENT_COL = {
    "S-shape": "dqn_improvement_vs_s_shape",
    "Return": "dqn_improvement_vs_return",
    "Largest Gap": "dqn_improvement_vs_largest_gap",
}


def parse_bool_series(series):
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


def load_results():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Final results not found:\n{INPUT_PATH}\n"
            "Run experiments.run_final_evaluation first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "distribution",
        "order_size",
        "dqn_completed",
        "dqn_distance",
        "dqn_gap",
        "dqn_collection_fraction",
        "dqn_invalid_action_rate",
        "exact_distance",
        "s_shape_distance",
        "return_distance",
        "largest_gap_distance",
        "dqn_improvement_vs_s_shape",
        "dqn_improvement_vs_return",
        "dqn_improvement_vs_largest_gap",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["dqn_completed"].dtype != bool:
        parsed = parse_bool_series(df["dqn_completed"])
        if parsed.isna().any():
            raise ValueError(
                "Could not parse some dqn_completed values."
            )
        df["dqn_completed"] = parsed.astype(bool)

    df["order_size"] = df["order_size"].astype(int)

    return df


def wilson_interval(successes, total):
    if total <= 0:
        return np.nan, np.nan

    alpha = 1.0 - CONFIDENCE_LEVEL
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = successes / total
    z2 = z * z

    denom = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denom
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z2 / (4.0 * total * total)
        )
        / denom
    )

    return max(0.0, centre - half), min(1.0, centre + half)


def holm_adjust(p_values):
    """
    Return Holm-adjusted p-values in original order.
    NaNs remain NaN.
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)

    valid_idx = np.where(np.isfinite(p_values))[0]
    if len(valid_idx) == 0:
        return adjusted

    ordered = valid_idx[
        np.argsort(p_values[valid_idx])
    ]

    m = len(ordered)
    running = 0.0

    for rank, idx in enumerate(ordered, start=1):
        candidate = min(
            1.0,
            (m - rank + 1) * p_values[idx],
        )
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)

    return adjusted


def completion_summary(df):
    rows = []

    for distribution in DISTRIBUTIONS:
        for size in ORDER_SIZES:
            sub = df[
                (df["distribution"] == distribution)
                & (df["order_size"] == size)
            ]

            n = len(sub)
            completed = int(sub["dqn_completed"].sum())
            failures = n - completed
            lower, upper = wilson_interval(completed, n)

            rows.append(
                {
                    "distribution": distribution,
                    "order_size": size,
                    "orders": n,
                    "completed": completed,
                    "failures": failures,
                    "completion_rate":
                        100.0 * completed / n,
                    "wilson_95_lower":
                        100.0 * lower,
                    "wilson_95_upper":
                        100.0 * upper,
                    "mean_collection_fraction":
                        sub["dqn_collection_fraction"].mean(),
                    "mean_invalid_action_rate":
                        sub["dqn_invalid_action_rate"].mean(),
                }
            )

    return pd.DataFrame(rows)


def overall_completion_summary(df):
    rows = []

    for distribution in DISTRIBUTIONS:
        sub = df[df["distribution"] == distribution]

        n = len(sub)
        completed = int(sub["dqn_completed"].sum())
        lower, upper = wilson_interval(completed, n)

        rows.append(
            {
                "distribution": distribution,
                "orders": n,
                "completed": completed,
                "failures": n - completed,
                "completion_rate":
                    100.0 * completed / n,
                "wilson_95_lower":
                    100.0 * lower,
                "wilson_95_upper":
                    100.0 * upper,
            }
        )

    return pd.DataFrame(rows)


def completion_distribution_tests(summary):
    rows = []

    for size in ORDER_SIZES:
        u = summary[
            (summary["distribution"] == "uniform")
            & (summary["order_size"] == size)
        ].iloc[0]

        c = summary[
            (summary["distribution"] == "clustered")
            & (summary["order_size"] == size)
        ].iloc[0]

        table = [
            [int(u["completed"]), int(u["failures"])],
            [int(c["completed"]), int(c["failures"])],
        ]

        result = fisher_exact(
            table,
            alternative="two-sided",
        )

        rows.append(
            {
                "order_size": size,
                "uniform_completed": int(u["completed"]),
                "uniform_failures": int(u["failures"]),
                "uniform_completion_rate":
                    float(u["completion_rate"]),
                "clustered_completed": int(c["completed"]),
                "clustered_failures": int(c["failures"]),
                "clustered_completion_rate":
                    float(c["completion_rate"]),
                "clustered_minus_uniform_pp":
                    float(
                        c["completion_rate"]
                        - u["completion_rate"]
                ),
                "fisher_odds_ratio":
                    float(result.statistic),
                "fisher_p_raw":
                    float(result.pvalue),
            }
        )

    out = pd.DataFrame(rows)

    out["fisher_p_holm"] = holm_adjust(
        out["fisher_p_raw"].to_numpy()
    )

    out["significant_holm_0_05"] = (
        out["fisher_p_holm"] < ALPHA
    )

    return out


def run_wilcoxon(differences):
    differences = np.asarray(
        differences,
        dtype=float,
    )
    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:
        return np.nan, np.nan, "no_pairs"

    nonzero = differences[
        np.abs(differences) > TOL
    ]

    if len(nonzero) == 0:
        return np.nan, np.nan, "all_pairs_equal_no_test"

    if len(nonzero) < 2:
        return np.nan, np.nan, "too_few_nonzero_pairs"

    try:
        result = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
    except TypeError:
        result = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            mode="auto",
        )

    return (
        float(result.statistic),
        float(result.pvalue),
        "performed",
    )


def paired_heuristic_tests(df):
    all_rows = []

    for distribution in DISTRIBUTIONS:
        for size in ORDER_SIZES:
            sub = df[
                (df["distribution"] == distribution)
                & (df["order_size"] == size)
                & (df["dqn_completed"])
                & (df["dqn_distance"].notna())
            ].copy()

            condition_rows = []

            for heuristic in HEURISTICS:
                h_col = DISTANCE_COL[heuristic]
                i_col = IMPROVEMENT_COL[heuristic]

                diff = (
                    sub["dqn_distance"]
                    - sub[h_col]
                ).to_numpy(dtype=float)

                wins = int(np.sum(diff < -TOL))
                ties = int(np.sum(np.abs(diff) <= TOL))
                losses = int(np.sum(diff > TOL))

                stat, p_raw, status = run_wilcoxon(diff)

                condition_rows.append(
                    {
                        "distribution": distribution,
                        "order_size": size,
                        "heuristic": heuristic,
                        "paired_n": len(sub),
                        "nonzero_pairs": wins + losses,
                        "dqn_wins": wins,
                        "ties": ties,
                        "dqn_losses": losses,
                        "dqn_win_rate_all_pairs":
                            (
                                100.0 * wins / len(sub)
                                if len(sub)
                                else np.nan
                        ),
                        "mean_dqn_minus_heuristic_distance":
                            float(np.mean(diff))
                            if len(diff)
                            else np.nan,
                        "median_dqn_minus_heuristic_distance":
                            float(np.median(diff))
                            if len(diff)
                            else np.nan,
                        "mean_dqn_improvement_percent":
                            float(sub[i_col].mean())
                            if len(sub)
                            else np.nan,
                        "wilcoxon_statistic": stat,
                        "wilcoxon_p_raw": p_raw,
                        "test_status": status,
                    }
                )

            temp = pd.DataFrame(condition_rows)

            temp["wilcoxon_p_holm"] = holm_adjust(
                temp["wilcoxon_p_raw"].to_numpy()
            )

            temp["significant_holm_0_05"] = (
                temp["wilcoxon_p_holm"] < ALPHA
            )

            all_rows.extend(
                temp.to_dict("records")
            )

    return pd.DataFrame(all_rows)


def paired_route_quality_summary(df):
    rows = []

    for distribution in DISTRIBUTIONS:
        for size in ORDER_SIZES:
            condition = df[
                (df["distribution"] == distribution)
                & (df["order_size"] == size)
            ]

            sub = condition[
                condition["dqn_completed"]
                & condition["dqn_distance"].notna()
            ].copy()

            def computed_gap(distance_col):
                return (
                    (sub[distance_col] - sub["exact_distance"])
                    / sub["exact_distance"]
                    * 100.0
                )

            rows.append(
                {
                    "distribution": distribution,
                    "order_size": size,
                    "condition_orders": len(condition),
                    "paired_route_quality_n": len(sub),
                    "dqn_completion_rate":
                        100.0 * len(sub) / len(condition),

                    "exact_mean_distance":
                        sub["exact_distance"].mean(),
                    "exact_median_distance":
                        sub["exact_distance"].median(),

                    "dqn_mean_distance":
                        sub["dqn_distance"].mean(),
                    "dqn_median_distance":
                        sub["dqn_distance"].median(),
                    "dqn_mean_gap":
                        sub["dqn_gap"].mean(),
                    "dqn_median_gap":
                        sub["dqn_gap"].median(),

                    "s_shape_mean_distance":
                        sub["s_shape_distance"].mean(),
                    "s_shape_median_distance":
                        sub["s_shape_distance"].median(),
                    "s_shape_mean_gap":
                        computed_gap(
                            "s_shape_distance"
                    ).mean(),

                    "return_mean_distance":
                        sub["return_distance"].mean(),
                    "return_median_distance":
                        sub["return_distance"].median(),
                    "return_mean_gap":
                        computed_gap(
                            "return_distance"
                    ).mean(),

                    "largest_gap_mean_distance":
                        sub["largest_gap_distance"].mean(),
                    "largest_gap_median_distance":
                        sub["largest_gap_distance"].median(),
                    "largest_gap_mean_gap":
                        computed_gap(
                            "largest_gap_distance"
                    ).mean(),
                }
            )

    return pd.DataFrame(rows)


def format_p(value):
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def build_text_summary(
    completion,
    overall,
    fisher_tests,
    paired_tests,
    route_quality,
):
    lines = []

    lines.append("FINAL RESULT STATISTICAL ANALYSIS")
    lines.append("=" * 72)
    lines.append("")

    lines.append("1. DQN COMPLETION BY CONDITION")
    lines.append("-" * 72)

    for _, row in completion.iterrows():
        lines.append(
            f"{row['distribution']:9s} | "
            f"{int(row['order_size']):2d} picks | "
            f"{int(row['completed'])}/{int(row['orders'])} "
            f"= {row['completion_rate']:.1f}% | "
            f"Wilson 95% CI "
            f"[{row['wilson_95_lower']:.1f}, "
            f"{row['wilson_95_upper']:.1f}]"
        )

    lines.append("")
    lines.append("Overall descriptive completion:")

    for _, row in overall.iterrows():
        lines.append(
            f"  {row['distribution']:9s}: "
            f"{int(row['completed'])}/{int(row['orders'])} "
            f"= {row['completion_rate']:.1f}% "
            f"(95% CI "
            f"{row['wilson_95_lower']:.1f}-"
            f"{row['wilson_95_upper']:.1f}%)"
        )

    lines.append("")
    lines.append("2. UNIFORM VS CLUSTERED COMPLETION")
    lines.append("-" * 72)
    lines.append(
        "Two-sided Fisher exact tests; "
        "Holm correction across 4 order sizes."
    )

    for _, row in fisher_tests.iterrows():
        lines.append(
            f"{int(row['order_size']):2d} picks | "
            f"uniform {row['uniform_completion_rate']:.1f}% | "
            f"clustered {row['clustered_completion_rate']:.1f}% | "
            f"change {row['clustered_minus_uniform_pp']:+.1f} pp | "
            f"p(raw)={format_p(row['fisher_p_raw'])} | "
            f"p(Holm)={format_p(row['fisher_p_holm'])} | "
            f"significant={bool(row['significant_holm_0_05'])}"
        )

    lines.append("")
    lines.append("3. PAIRED DQN VS HEURISTICS")
    lines.append("-" * 72)
    lines.append(
        "Completed DQN orders only. "
        "Difference = DQN - heuristic; positive means DQN is longer."
    )
    lines.append(
        "Wilcoxon tests are two-sided with Holm correction "
        "across the 3 heuristics within each condition."
    )

    for distribution in DISTRIBUTIONS:
        for size in ORDER_SIZES:
            lines.append("")
            lines.append(
                f"{distribution.upper()} | {size} PICKS"
            )

            subset = paired_tests[
                (paired_tests["distribution"] == distribution)
                & (paired_tests["order_size"] == size)
            ]

            for _, row in subset.iterrows():
                lines.append(
                    f"  {row['heuristic']:11s} | "
                    f"n={int(row['paired_n']):2d} | "
                    f"W/T/L="
                    f"{int(row['dqn_wins'])}/"
                    f"{int(row['ties'])}/"
                    f"{int(row['dqn_losses'])} | "
                    f"mean diff="
                    f"{row['mean_dqn_minus_heuristic_distance']:+.2f} | "
                    f"median diff="
                    f"{row['median_dqn_minus_heuristic_distance']:+.2f} | "
                    f"mean improvement="
                    f"{row['mean_dqn_improvement_percent']:+.2f}% | "
                    f"p(Holm)="
                    f"{format_p(row['wilcoxon_p_holm'])} | "
                    f"{row['test_status']}"
                )

    lines.append("")
    lines.append("4. PAIRED ROUTE-QUALITY GAPS")
    lines.append("-" * 72)
    lines.append(
        "Every method below is restricted to the same completed-DQN orders."
    )

    for _, row in route_quality.iterrows():
        lines.append(
            f"{row['distribution']:9s} | "
            f"{int(row['order_size']):2d} picks | "
            f"n={int(row['paired_route_quality_n']):2d} | "
            f"DQN={row['dqn_mean_gap']:.2f}% | "
            f"S={row['s_shape_mean_gap']:.2f}% | "
            f"R={row['return_mean_gap']:.2f}% | "
            f"LG={row['largest_gap_mean_gap']:.2f}%"
        )

    lines.append("")
    lines.append("INTERPRETATION NOTES")
    lines.append("-" * 72)
    lines.append(
        "* Incomplete DQN partial distance is not a valid route solution."
    )
    lines.append(
        "* Direct route-quality comparisons use completed DQN orders only."
    )
    lines.append(
        "* Lower OOD optimality gaps cannot alone imply better "
        "generalisation because the successful subset changes with completion."
    )
    lines.append(
        "* Equal DQN/Return distances do not prove identical paths."
    )

    return "\n".join(lines)


def save_config():
    config = {
        "input": str(INPUT_PATH),
        "alpha": ALPHA,
        "confidence_level": CONFIDENCE_LEVEL,
        "completion_interval": "Wilson score interval",
        "completion_distribution_test":
            "two-sided Fisher exact",
        "completion_multiple_testing":
            "Holm across 4 order sizes",
        "paired_route_test":
            "two-sided Wilcoxon signed-rank",
        "paired_route_multiple_testing":
            "Holm across 3 heuristics within each condition",
        "paired_route_subset":
            "completed DQN orders only",
        "paired_difference":
            "DQN distance - heuristic distance",
        "order_sizes": ORDER_SIZES,
        "distributions": DISTRIBUTIONS,
        "heuristics": HEURISTICS,
        "additional_training": False,
        "additional_simulation": False,
    }

    with (
        ANALYSIS_DIR / "analysis_config.json"
    ).open("w") as f:
        json.dump(config, f, indent=4)


def main():
    print(
        "\n"
        "================================================\n"
        "ANALYZE FINAL WAREHOUSE ROUTING RESULTS\n"
        "================================================"
    )

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results()

    print(f"Loaded {len(df)} final-test rows.")

    completion = completion_summary(df)
    overall = overall_completion_summary(df)
    fisher_tests = completion_distribution_tests(
        completion
    )
    paired_tests = paired_heuristic_tests(df)
    route_quality = paired_route_quality_summary(
        df
    )

    completion.to_csv(
        ANALYSIS_DIR / "completion_summary.csv",
        index=False,
    )

    overall.to_csv(
        ANALYSIS_DIR / "overall_completion_summary.csv",
        index=False,
    )

    fisher_tests.to_csv(
        ANALYSIS_DIR / "completion_distribution_tests.csv",
        index=False,
    )

    paired_tests.to_csv(
        ANALYSIS_DIR / "paired_heuristic_tests.csv",
        index=False,
    )

    route_quality.to_csv(
        ANALYSIS_DIR / "paired_route_quality_summary.csv",
        index=False,
    )

    save_config()

    report = build_text_summary(
        completion,
        overall,
        fisher_tests,
        paired_tests,
        route_quality,
    )

    (
        ANALYSIS_DIR
        / "analysis_summary.txt"
    ).write_text(
        report + "\n"
    )

    print("\n" + report)

    print(
        "\n"
        "================================================\n"
        "FINAL STATISTICAL ANALYSIS COMPLETE\n"
        "================================================"
    )

    print(f"\nSaved to:\n  {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
