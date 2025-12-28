#!/usr/bin/env python3
"""
Analysis Reporting Module

Handles statistical analysis and reporting of test generation results,
including dataset-aware analysis by complexity and algorithm types.
"""

from typing import List, Dict, Any
import pandas as pd


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
        self.print_ast_fix_analysis()
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

    def print_ast_fix_analysis(self) -> None:
        """Print analysis comparing runs with and without ast-fix option.

        Robust to missing optional columns like fix_attempts_used, total_cost_usd,
        and code_coverage_percent.
        """
        if "has_ast_fix" not in self.df.columns or "success" not in self.df.columns:
            return

        print("\n" + "=" * 80)
        print("AST-FIX ANALYSIS")
        print("=" * 80)

        # Build aggregation dict based on available columns
        agg_dict = {
            "success_rate": ("success", "mean"),
            "samples": ("success", "count"),
        }
        if "fix_attempts_used" in self.df.columns:
            agg_dict["avg_fix_attempts"] = ("fix_attempts_used", "mean")
        if "total_cost_usd" in self.df.columns:
            agg_dict["avg_cost_usd"] = ("total_cost_usd", "mean")
        if "code_coverage_percent" in self.df.columns:
            agg_dict["avg_coverage"] = ("code_coverage_percent", "mean")

        grouped = (
            self.df.groupby(["has_ast_fix", "config_type"], observed=False)
            .agg(**agg_dict)
            .reset_index()
        )

        if grouped.empty:
            print("No data available for ast-fix analysis.")
            return

        # Pretty print
        for has_ast_fix in [False, True]:
            subset = grouped[grouped["has_ast_fix"] == has_ast_fix]
            label = "WITH ast-fix" if has_ast_fix else "WITHOUT ast-fix"
            print(f"\n{label}:")
            for _, row in subset.iterrows():
                fields = [
                    f"success={row['success_rate']*100:.1f}% (n={int(row['samples'])})",
                ]
                if "avg_fix_attempts" in row.index:
                    fields.append(f"fix_attempts={row['avg_fix_attempts']:.2f}")
                if "avg_cost_usd" in row.index:
                    fields.append(f"cost=${row['avg_cost_usd']:.4f}")
                if "avg_coverage" in row.index:
                    fields.append(f"coverage={row['avg_coverage']:.1f}%")

                print(f"  {row['config_type']}: " + ", ".join(fields))

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
