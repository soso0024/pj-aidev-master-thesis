#!/usr/bin/env python3
"""
Analysis Reporting Module

Handles statistical analysis and reporting of test generation results,
including dataset-aware analysis by complexity and algorithm types.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Efficiency Metric Constants
# C1 (Branch Coverage) is weighted higher because achieving C1 implies C0 coverage
# Theoretical basis: C1 ⊃ C0 (branch coverage subsumes statement coverage)
C0_WEIGHT = 0.3  # Statement Coverage weight
C1_WEIGHT = 0.7  # Branch Coverage weight (higher because C1 > C0)
COST_MULTIPLIER = 1000  # Normalize cost to per $0.001


class AnalysisReporter:
    """Handles statistical analysis and reporting of test result data."""

    def __init__(self, data: List[Dict[str, Any]], config_order: List[str]):
        """Initialize with loaded data and configuration order."""
        self.data = data
        self.config_order = config_order
        self.df = self._prepare_dataframe()

    def _prepare_dataframe(self) -> pd.DataFrame:
        """Prepare pandas DataFrame from loaded data."""
        df = pd.DataFrame(self.data)

        # Create ordered categorical for consistent display
        df["config_type_display"] = pd.Categorical(
            df["config_type"], categories=self.config_order, ordered=True
        )

        return df

    def generate_all_reports(self) -> None:
        """Generate all analysis reports."""
        print("Generating analysis reports...")
        self.print_summary_stats()
        self.print_efficiency_analysis()
        print("\nAnalysis reports completed!")

    def print_summary_stats(self) -> None:
        """Print summary statistics for all configurations."""
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return

        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        summary = (
            self.df.groupby("config_type")
            .agg(
                {
                    "success": ["count", "mean"],
                    "total_input_tokens": ["mean", "std"],
                    "total_cost_usd": ["mean", "std"],
                    "code_coverage_percent": ["mean", "std"],
                    "fix_attempts_used": ["mean", "std"],
                }
            )
            .round(3)
        )

        print(summary)

        print("\n" + "=" * 80)
        print("CONFIGURATION TYPE ANALYSIS")
        print("=" * 80)

        for config_type in sorted(self.df["config_type"].unique()):
            data = self.df[self.df["config_type"] == config_type]
            print(f"\n{config_type.upper()}:")
            print(f"  Samples: {len(data)}")
            print(f"  Success Rate: {data['success'].mean()*100:.1f}%")
            print(
                f"  Avg Input Tokens: {data['total_input_tokens'].mean():.0f} ± {data['total_input_tokens'].std():.0f}"
            )
            print(
                f"  Avg Cost: ${data['total_cost_usd'].mean():.4f} ± ${data['total_cost_usd'].std():.4f}"
            )
            print(
                f"  Avg Coverage: {data['code_coverage_percent'].mean():.1f}% ± {data['code_coverage_percent'].std():.1f}%"
            )
            print(
                f"  Avg Fix Attempts: {data['fix_attempts_used'].mean():.2f} ± {data['fix_attempts_used'].std():.2f}"
            )

        # Add dataset-aware analysis
        self.print_dataset_analysis()

    def print_dataset_analysis(self) -> None:
        """Print dataset-aware analysis including complexity and algorithm type breakdown."""
        if "complexity_level" not in self.df.columns:
            print("\nDataset classification not available.")
            return

        print("\n" + "=" * 80)
        print("DATASET COMPLEXITY ANALYSIS")
        print("=" * 80)

        # Overall complexity distribution
        complexity_dist = self.df["complexity_level"].value_counts()
        print("\nProblem Complexity Distribution:")
        for level, count in complexity_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {level.capitalize()}: {count} problems ({percentage:.1f}%)")

        # Success rates by complexity
        print("\nSuccess Rate by Complexity Level:")
        complexity_success = (
            self.df.groupby(
                ["complexity_level", "config_type_display"], observed=False
            )["success"]
            .agg(["mean", "count"])
            .reset_index()
        )
        complexity_success["success_rate"] = complexity_success["mean"] * 100

        for complexity in ["simple", "medium", "complex"]:
            complexity_data = complexity_success[
                complexity_success["complexity_level"] == complexity
            ]
            if not complexity_data.empty:
                print(f"\n  {complexity.capitalize()} Problems:")
                for _, row in complexity_data.iterrows():
                    config = row["config_type_display"]
                    rate = row["success_rate"]
                    count = row["count"]
                    print(f"    {config}: {rate:.1f}% ({count} samples)")

        # Algorithm type analysis
        if "algorithm_type" in self.df.columns:
            print("\n" + "=" * 80)
            print("ALGORITHM TYPE ANALYSIS")
            print("=" * 80)

            # Algorithm type distribution
            algo_dist = self.df["algorithm_type"].value_counts()
            print("\nAlgorithm Type Distribution:")
            for algo_type, count in algo_dist.head(10).items():  # Show top 10
                percentage = (count / len(self.df)) * 100
                print(
                    f"  {algo_type.replace('_', ' ').title()}: {count} problems ({percentage:.1f}%)"
                )

            # Success rates by algorithm type (for types with sufficient data)
            algo_success = (
                self.df.groupby("algorithm_type")["success"]
                .agg(["mean", "count"])
                .reset_index()
            )
            algo_success["success_rate"] = algo_success["mean"] * 100
            sufficient_algos = algo_success[algo_success["count"] >= 3].sort_values(
                "success_rate", ascending=False
            )

            if not sufficient_algos.empty:
                print("\nSuccess Rate by Algorithm Type (≥3 samples):")
                for _, row in sufficient_algos.iterrows():
                    algo = row["algorithm_type"].replace("_", " ").title()
                    rate = row["success_rate"]
                    count = row["count"]
                    print(f"  {algo}: {rate:.1f}% ({count} samples)")

        # Configuration effectiveness analysis
        print("\n" + "=" * 80)
        print("CONFIGURATION EFFECTIVENESS BY PROBLEM TYPE")
        print("=" * 80)

        # Best configuration for each complexity level
        if "complexity_level" in self.df.columns:
            print("\nBest Configuration by Complexity Level:")
            for complexity in ["simple", "medium", "complex"]:
                complexity_data = self.df[self.df["complexity_level"] == complexity]
                if not complexity_data.empty:
                    config_performance = (
                        complexity_data.groupby("config_type_display")["success"]
                        .agg(["mean", "count"])
                        .reset_index()
                    )
                    config_performance = config_performance[
                        config_performance["count"] >= 2
                    ]  # At least 2 samples
                    if not config_performance.empty:
                        best_config = config_performance.loc[
                            config_performance["mean"].idxmax()
                        ]
                        print(
                            f"  {complexity.capitalize()}: {best_config['config_type_display']} "
                            + f"({best_config['mean']*100:.1f}% success rate, {best_config['count']} samples)"
                        )
                    else:
                        print(f"  {complexity.capitalize()}: Insufficient data")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics as dictionary."""
        if not self.data:
            return {}

        summary = (
            self.df.groupby("config_type")
            .agg(
                {
                    "success": ["count", "mean"],
                    "total_input_tokens": ["mean", "std"],
                    "total_cost_usd": ["mean", "std"],
                    "code_coverage_percent": ["mean", "std"],
                    "fix_attempts_used": ["mean", "std"],
                }
            )
            .round(3)
        )

        return summary.to_dict()

    def print_efficiency_analysis(self) -> None:
        """Print efficiency metrics analysis focusing on SCCE.

        Metric:
        - SCCE: Success × (0.3×C0 + 0.7×C1) / (Cost × 1000) - Success-weighted efficiency

        C1 is weighted higher (0.7) because branch coverage (C1) subsumes statement
        coverage (C0) - achieving C1 implies C0 is also achieved.
        """
        if not self.data:
            print("No data loaded for efficiency analysis.")
            return

        print("\n" + "=" * 80)
        print("EFFICIENCY METRICS ANALYSIS")
        print("=" * 80)
        print("\nMetric Definition:")
        print(
            f"  SCCE: Success × ({C0_WEIGHT}×C0 + {C1_WEIGHT}×C1) / (Cost × {COST_MULTIPLIER})"
        )
        print(
            f"  Note: C1 weight ({C1_WEIGHT}) > C0 weight ({C0_WEIGHT}) because C1 ⊃ C0"
        )

        # Ensure C0/C1 columns exist with fallback
        if "code_coverage_c0_percent" not in self.df.columns:
            self.df["code_coverage_c0_percent"] = self.df["code_coverage_percent"]
        if "code_coverage_c1_percent" not in self.df.columns:
            self.df["code_coverage_c1_percent"] = 100.0

        print("\n" + "-" * 80)
        print("EFFICIENCY BY CONFIGURATION TYPE")
        print("-" * 80)

        for config_type in sorted(self.df["config_type"].unique()):
            data = self.df[self.df["config_type"] == config_type]

            avg_cost = data["total_cost_usd"].mean()
            avg_c0 = data["code_coverage_c0_percent"].mean()
            avg_c1 = data["code_coverage_c1_percent"].mean()
            success_rate = data["success"].mean()
            samples = len(data)

            # Skip if cost is zero to avoid division by zero
            if avg_cost <= 0:
                print(f"\n{config_type.upper()}: (skipped - no cost data)")
                continue

            # Calculate efficiency metric
            weighted_coverage = C0_WEIGHT * avg_c0 + C1_WEIGHT * avg_c1
            scce = (success_rate * weighted_coverage) / (avg_cost * COST_MULTIPLIER)

            print(f"\n{config_type.upper()} (n={samples}):")
            print(f"  Inputs:")
            print(f"    Avg Cost:         ${avg_cost:.6f}")
            print(f"    Avg C0 Coverage:  {avg_c0:.1f}%")
            print(f"    Avg C1 Coverage:  {avg_c1:.1f}%")
            print(f"    Success Rate:     {success_rate*100:.1f}%")
            print(f"  Efficiency Score:")
            print(f"    SCCE:             {scce:.2f}")

        # Print ranking by SCCE
        print("\n" + "-" * 80)
        print("EFFICIENCY RANKING BY SCCE")
        print("-" * 80)

        rankings = []
        for config_type in self.df["config_type"].unique():
            data = self.df[self.df["config_type"] == config_type]
            avg_cost = data["total_cost_usd"].mean()

            if avg_cost <= 0:
                continue

            avg_c0 = data["code_coverage_c0_percent"].mean()
            avg_c1 = data["code_coverage_c1_percent"].mean()
            success_rate = data["success"].mean()
            weighted_coverage = C0_WEIGHT * avg_c0 + C1_WEIGHT * avg_c1
            scce = (success_rate * weighted_coverage) / (avg_cost * COST_MULTIPLIER)

            rankings.append(
                {
                    "config": config_type,
                    "scce": scce,
                    "success_rate": success_rate * 100,
                    "weighted_cov": weighted_coverage,
                    "cost": avg_cost,
                }
            )

        # Sort by SCCE descending
        rankings.sort(key=lambda x: x["scce"], reverse=True)

        print("\nRank | Configuration    | SCCE   | Success | W.Coverage | Avg Cost")
        print("-" * 75)
        for rank, r in enumerate(rankings, 1):
            print(
                f"  {rank:2d} | {r['config']:<16} | {r['scce']:6.2f} | {r['success_rate']:5.1f}%  | "
                f"{r['weighted_cov']:6.1f}%    | ${r['cost']:.6f}"
            )
