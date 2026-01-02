#!/usr/bin/env python3
"""
Analysis Reporting Module

Handles statistical analysis and reporting of test generation results,
including dataset-aware analysis by complexity and algorithm types.
Includes statistical significance testing for research papers.
"""

from typing import List, Dict, Any, Tuple, Optional
from itertools import combinations
import pandas as pd
import numpy as np
from scipy import stats

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
        self.print_ast_fix_analysis()
        self.print_efficiency_analysis()
        self.print_statistical_significance_tests()
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

    def print_efficiency_analysis(self) -> None:
        """Print efficiency metrics analysis including CCE-C0, CCE-C1, and SCCE.

        Metrics:
        - CCE-C0: C0 Coverage / (Cost × 1000) - Statement coverage efficiency
        - CCE-C1: C1 Coverage / (Cost × 1000) - Branch coverage efficiency
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
        print("\nMetric Definitions:")
        print(f"  CCE-C0: C0 Coverage / (Cost × {COST_MULTIPLIER}) - Statement coverage efficiency")
        print(f"  CCE-C1: C1 Coverage / (Cost × {COST_MULTIPLIER}) - Branch coverage efficiency")
        print(
            f"  SCCE:   Success × ({C0_WEIGHT}×C0 + {C1_WEIGHT}×C1) / (Cost × {COST_MULTIPLIER})"
        )
        print(f"\n  Note: C1 weight ({C1_WEIGHT}) > C0 weight ({C0_WEIGHT}) because C1 ⊃ C0")

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

            # Calculate efficiency metrics
            cce_c0 = avg_c0 / (avg_cost * COST_MULTIPLIER)
            cce_c1 = avg_c1 / (avg_cost * COST_MULTIPLIER)
            weighted_coverage = C0_WEIGHT * avg_c0 + C1_WEIGHT * avg_c1
            scce = (success_rate * weighted_coverage) / (avg_cost * COST_MULTIPLIER)

            print(f"\n{config_type.upper()} (n={samples}):")
            print(f"  Inputs:")
            print(f"    Avg Cost:         ${avg_cost:.6f}")
            print(f"    Avg C0 Coverage:  {avg_c0:.1f}%")
            print(f"    Avg C1 Coverage:  {avg_c1:.1f}%")
            print(f"    Success Rate:     {success_rate*100:.1f}%")
            print(f"  Efficiency Scores:")
            print(f"    CCE-C0:           {cce_c0:.2f}")
            print(f"    CCE-C1:           {cce_c1:.2f}")
            print(f"    SCCE:             {scce:.2f} ⭐ (Success-weighted)")

        # Print ranking by SCCE
        print("\n" + "-" * 80)
        print("EFFICIENCY RANKING BY SCCE (Success-weighted)")
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

    # ========================================================================
    # STATISTICAL SIGNIFICANCE TESTING
    # ========================================================================

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size between two groups.
        
        Cohen's d interpretation:
        - |d| < 0.2: negligible effect
        - 0.2 ≤ |d| < 0.5: small effect
        - 0.5 ≤ |d| < 0.8: medium effect
        - |d| ≥ 0.8: large effect
        """
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return np.nan
        
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def _cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size).
        
        Cliff's delta interpretation:
        - |δ| < 0.147: negligible
        - 0.147 ≤ |δ| < 0.33: small
        - 0.33 ≤ |δ| < 0.474: medium
        - |δ| ≥ 0.474: large
        """
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return np.nan
        
        # Count dominance
        greater = sum(1 for x in group1 for y in group2 if x > y)
        less = sum(1 for x in group1 for y in group2 if x < y)
        
        return (greater - less) / (n1 * n2)

    def _interpret_effect_size(self, d: float, metric: str = "cohens_d") -> str:
        """Interpret effect size magnitude."""
        if np.isnan(d):
            return "N/A"
        
        abs_d = abs(d)
        
        if metric == "cohens_d":
            if abs_d < 0.2:
                return "negligible"
            elif abs_d < 0.5:
                return "small"
            elif abs_d < 0.8:
                return "medium"
            else:
                return "large"
        else:  # Cliff's delta
            if abs_d < 0.147:
                return "negligible"
            elif abs_d < 0.33:
                return "small"
            elif abs_d < 0.474:
                return "medium"
            else:
                return "large"

    def _bootstrap_ci(
        self, data: np.ndarray, statistic: str = "mean", 
        confidence: float = 0.95, n_iterations: int = 10000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval.
        
        Args:
            data: Input data array
            statistic: "mean" or "median"
            confidence: Confidence level (default 95%)
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(data) < 2:
            return (np.nan, np.nan)
        
        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_iterations):
            sample = np.random.choice(data, size=len(data), replace=True)
            if statistic == "mean":
                bootstrap_stats.append(np.mean(sample))
            else:
                bootstrap_stats.append(np.median(sample))
        
        # Calculate percentiles
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        
        return (lower, upper)

    def _format_p_value(self, p: float) -> str:
        """Format p-value for display with significance stars."""
        if np.isnan(p):
            return "N/A"
        if p < 0.001:
            return f"{p:.2e} ***"
        elif p < 0.01:
            return f"{p:.4f} **"
        elif p < 0.05:
            return f"{p:.4f} *"
        else:
            return f"{p:.4f}"

    def print_statistical_significance_tests(self) -> None:
        """Print comprehensive statistical significance tests for research papers.
        
        Includes:
        - Kruskal-Wallis H-test (overall group comparison)
        - Pairwise Mann-Whitney U tests with Bonferroni correction
        - Effect sizes (Cohen's d and Cliff's delta)
        - 95% Confidence Intervals via bootstrap
        """
        if not self.data or len(self.df) < 2:
            print("\nInsufficient data for statistical significance testing.")
            return

        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print("=" * 80)
        print("\nNote: Statistical tests for research paper validation")
        print("  * p < 0.05, ** p < 0.01, *** p < 0.001")
        
        config_types = sorted(self.df["config_type"].unique())
        
        if len(config_types) < 2:
            print("\nNeed at least 2 configuration types for comparison.")
            return

        # Metrics to analyze
        metrics = [
            ("success", "Success Rate"),
            ("code_coverage_percent", "Code Coverage (%)"),
            ("total_cost_usd", "Total Cost (USD)"),
            ("fix_attempts_used", "Fix Attempts"),
        ]
        
        # Add C0/C1 if available
        if "code_coverage_c0_percent" in self.df.columns:
            metrics.append(("code_coverage_c0_percent", "C0 Coverage (%)"))
        if "code_coverage_c1_percent" in self.df.columns:
            metrics.append(("code_coverage_c1_percent", "C1 Coverage (%)"))

        for metric_col, metric_name in metrics:
            if metric_col not in self.df.columns:
                continue
                
            print(f"\n{'─' * 80}")
            print(f"📊 {metric_name}")
            print("─" * 80)
            
            # Prepare data for each configuration
            groups = {}
            for config in config_types:
                data = self.df[self.df["config_type"] == config][metric_col].dropna()
                if len(data) > 0:
                    groups[config] = data.values

            if len(groups) < 2:
                print("  Insufficient data for comparison.")
                continue

            # ================================================================
            # 1. DESCRIPTIVE STATISTICS WITH 95% CI
            # ================================================================
            print("\n  📈 Descriptive Statistics (with 95% CI):")
            print("  " + "-" * 70)
            print(f"  {'Config':<15} | {'Mean':>10} | {'Std':>8} | {'95% CI':>20} | {'n':>5}")
            print("  " + "-" * 70)
            
            for config, data in groups.items():
                mean = np.mean(data)
                std = np.std(data, ddof=1) if len(data) > 1 else 0
                ci_low, ci_high = self._bootstrap_ci(data, "mean", 0.95, 5000)
                n = len(data)
                
                # Format based on metric type
                if metric_col == "total_cost_usd":
                    print(f"  {config:<15} | ${mean:>9.6f} | ${std:>7.6f} | "
                          f"[${ci_low:.6f}, ${ci_high:.6f}] | {n:>5}")
                elif "percent" in metric_col or metric_col == "success":
                    display_mean = mean * 100 if metric_col == "success" else mean
                    display_std = std * 100 if metric_col == "success" else std
                    display_ci_low = ci_low * 100 if metric_col == "success" else ci_low
                    display_ci_high = ci_high * 100 if metric_col == "success" else ci_high
                    print(f"  {config:<15} | {display_mean:>9.1f}% | {display_std:>7.1f}% | "
                          f"[{display_ci_low:.1f}%, {display_ci_high:.1f}%] | {n:>5}")
                else:
                    print(f"  {config:<15} | {mean:>10.2f} | {std:>8.2f} | "
                          f"[{ci_low:.2f}, {ci_high:.2f}] | {n:>5}")

            # ================================================================
            # 2. KRUSKAL-WALLIS H-TEST (Overall comparison)
            # ================================================================
            if len(groups) >= 2:
                print("\n  🔬 Kruskal-Wallis H-test (Overall Comparison):")
                try:
                    group_arrays = list(groups.values())
                    h_stat, p_value = stats.kruskal(*group_arrays)
                    print(f"      H-statistic: {h_stat:.4f}")
                    print(f"      p-value:     {self._format_p_value(p_value)}")
                    
                    if p_value < 0.05:
                        print("      → Significant difference exists between groups")
                    else:
                        print("      → No significant difference between groups")
                except Exception as e:
                    print(f"      Could not compute: {e}")

            # ================================================================
            # 3. PAIRWISE MANN-WHITNEY U TESTS WITH BONFERRONI CORRECTION
            # ================================================================
            config_pairs = list(combinations(groups.keys(), 2))
            n_comparisons = len(config_pairs)
            bonferroni_alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
            
            if n_comparisons > 0:
                print(f"\n  🔍 Pairwise Mann-Whitney U Tests (Bonferroni α = {bonferroni_alpha:.4f}):")
                print("  " + "-" * 70)
                print(f"  {'Comparison':<25} | {'U-stat':>10} | {'p-value':>15} | {'Significant':>10}")
                print("  " + "-" * 70)
                
                pairwise_results = []
                for config1, config2 in config_pairs:
                    try:
                        u_stat, p_value = stats.mannwhitneyu(
                            groups[config1], groups[config2], alternative='two-sided'
                        )
                        significant = "Yes" if p_value < bonferroni_alpha else "No"
                        pairwise_results.append({
                            "pair": f"{config1} vs {config2}",
                            "u_stat": u_stat,
                            "p_value": p_value,
                            "significant": significant,
                            "config1": config1,
                            "config2": config2
                        })
                        print(f"  {config1} vs {config2:<10} | {u_stat:>10.1f} | "
                              f"{self._format_p_value(p_value):>15} | {significant:>10}")
                    except Exception as e:
                        print(f"  {config1} vs {config2:<10} | Could not compute: {e}")

            # ================================================================
            # 4. EFFECT SIZES
            # ================================================================
            if n_comparisons > 0:
                print("\n  📏 Effect Sizes:")
                print("  " + "-" * 70)
                cohens_label = "Cohen's d"
                cliffs_label = "Cliff's δ"
                print(f"  {'Comparison':<25} | {cohens_label:>10} | {'Interpretation':>12} | "
                      f"{cliffs_label:>10} | {'Interpretation':>12}")
                print("  " + "-" * 70)
                
                for config1, config2 in config_pairs:
                    d = self._cohens_d(groups[config1], groups[config2])
                    delta = self._cliffs_delta(groups[config1], groups[config2])
                    
                    d_interp = self._interpret_effect_size(d, "cohens_d")
                    delta_interp = self._interpret_effect_size(delta, "cliffs_delta")
                    
                    d_str = f"{d:.3f}" if not np.isnan(d) else "N/A"
                    delta_str = f"{delta:.3f}" if not np.isnan(delta) else "N/A"
                    
                    print(f"  {config1} vs {config2:<10} | {d_str:>10} | {d_interp:>12} | "
                          f"{delta_str:>10} | {delta_interp:>12}")

        # ================================================================
        # SUMMARY FOR PAPER
        # ================================================================
        print("\n" + "=" * 80)
        print("📝 SUMMARY FOR RESEARCH PAPER")
        print("=" * 80)
        print("""
Statistical analysis was performed using non-parametric tests due to 
the nature of the data. The Kruskal-Wallis H-test was used for overall
group comparisons, followed by pairwise Mann-Whitney U tests with 
Bonferroni correction for multiple comparisons. Effect sizes were 
calculated using both Cohen's d (parametric) and Cliff's delta 
(non-parametric) to quantify the magnitude of observed differences.
95% confidence intervals were estimated using bootstrap resampling 
(n=5,000 iterations).

Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001

Effect size interpretation (Cohen's d):
  - |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, ≥0.8: large

Effect size interpretation (Cliff's δ):
  - |δ| < 0.147: negligible, 0.147-0.33: small, 0.33-0.474: medium, ≥0.474: large
""")
