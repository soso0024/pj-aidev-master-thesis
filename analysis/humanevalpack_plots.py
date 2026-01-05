#!/usr/bin/env python3
"""
HumanEvalPack-Specific Plotting Module

Contains visualization methods specifically designed for analyzing
bug detection results from the HumanEvalPack dataset.

Key metrics visualized:
- Bug detection rates by bug type
- True Positive vs False Positive analysis
- Failure type distribution
- Bug detection success by model and configuration
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class HumanEvalPackPlots:
    """Handles HumanEvalPack-specific visualization methods."""

    # Bug type display names
    BUG_TYPE_DISPLAY = {
        "missing logic": "Missing Logic",
        "excess logic": "Excess Logic",
        "missing condition": "Missing Condition",
        "operator misuse": "Operator Misuse",
        "value misuse": "Value Misuse",
        "variable misuse": "Variable Misuse",
        "function misuse": "Function Misuse",
    }

    def __init__(
        self,
        data: List[Dict[str, Any]],
        config_order: List[str],
    ):
        """
        Initialize with loaded data and ordering configurations.
        
        Args:
            data: List of test result dictionaries
            config_order: Ordered list of configuration types
        """
        self.data = data
        self.config_order = config_order
        self.df = self._prepare_dataframe()

    def _prepare_dataframe(self) -> pd.DataFrame:
        """Prepare pandas DataFrame from loaded data, filtering HumanEvalPack only."""
        df = pd.DataFrame(self.data)
        
        # Filter to HumanEvalPack data only
        if "dataset_type" in df.columns:
            df = df[df["dataset_type"] == "humanevalpack"]
        
        if df.empty:
            return df
        
        # Create ordered categorical for consistent display
        if "config_type" in df.columns:
            df["config_type_display"] = pd.Categorical(
                df["config_type"], categories=self.config_order, ordered=True
            )
        
        return df

    def _format_model_name(self, model_name: str) -> str:
        """Format model name for display."""
        model_display_names = {
            "claude-3-5-haiku": "Claude 3.5 Haiku",
            "claude-opus-4-1": "Claude Opus 4.1",
            "claude-4-sonnet": "Claude 4 Sonnet",
            "claude-3-haiku": "Claude 3 Haiku",
            "claude-haiku-4-5": "Claude Haiku 4.5",
            "claude-4-5-sonnet": "Claude 4.5 Sonnet",
        }
        return model_display_names.get(model_name, model_name.replace("-", " ").title())

    def has_data(self) -> bool:
        """Check if there is HumanEvalPack data available."""
        return not self.df.empty

    def create_all_plots(self, output_path: Path) -> None:
        """Create all HumanEvalPack-specific visualization plots."""
        if not self.has_data():
            print("⚠️  No HumanEvalPack data available. Skipping HumanEvalPack plots.")
            return

        print("Creating HumanEvalPack-specific visualizations...")

        self._plot_bug_detection_by_type(output_path)
        print("  ✓ Created bug detection by type heatmap")

        self._plot_true_vs_false_positive(output_path)
        print("  ✓ Created True vs False Positive comparison")

        self._plot_failure_type_distribution(output_path)
        print("  ✓ Created failure type distribution")

        self._plot_bug_detection_by_model(output_path)
        print("  ✓ Created bug detection by model analysis")

        self._plot_detection_success_summary(output_path)
        print("  ✓ Created detection success summary")

    def _plot_bug_detection_by_type(self, output_path: Path) -> None:
        """Plot bug detection rate heatmap by bug type and configuration."""
        if "bug_type" not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate detection rates by bug type and configuration
        detection_rates = self.df.pivot_table(
            values="is_true_positive",
            index="bug_type",
            columns="config_type_display",
            aggfunc=lambda x: (x == True).mean() * 100 if len(x) > 0 else 0,
            fill_value=0,
            observed=False,
        )

        if detection_rates.empty:
            ax.text(0.5, 0.5, "No bug detection data available",
                   ha="center", va="center", transform=ax.transAxes, fontsize=14)
        else:
            # Rename index for display
            detection_rates.index = detection_rates.index.map(
                lambda x: self.BUG_TYPE_DISPLAY.get(x, x.replace("_", " ").title())
            )

            heatmap = sns.heatmap(
                detection_rates,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn",
                vmin=0,
                vmax=100,
                ax=ax,
                cbar_kws={"label": "True Bug Detection Rate (%)"},
                annot_kws={"fontsize": 14, "fontweight": "bold"},
            )
            
            # Style colorbar
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label("True Bug Detection Rate (%)", size=14)

        ax.set_xlabel("Configuration Type", fontsize=16, fontweight="bold")
        ax.set_ylabel("Bug Type", fontsize=16, fontweight="bold")
        ax.set_title("Bug Detection Rate by Bug Type and Configuration", 
                    fontsize=18, fontweight="bold", pad=20)
        ax.tick_params(axis="both", which="major", labelsize=12)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(
            output_path / "9_bug_detection_by_type.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_true_vs_false_positive(self, output_path: Path) -> None:
        """Plot comparison of True Positive vs False Positive rates."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Calculate rates by configuration
        config_stats = []
        for config in self.df["config_type"].unique():
            config_data = self.df[self.df["config_type"] == config]
            total = len(config_data)
            
            if total == 0:
                continue
            
            # Calculate metrics
            true_positives = (config_data["is_true_positive"] == True).sum()
            false_positives = (config_data["is_false_positive"] == True).sum()
            
            # Bug detection success = canonical passed AND buggy failed
            canonical_passed = (config_data["canonical_solution_passed"] == True).sum()
            buggy_failed = (config_data["buggy_solution_failed"] == True).sum()
            
            config_stats.append({
                "config": config,
                "true_positive_rate": (true_positives / total) * 100,
                "false_positive_rate": (false_positives / total) * 100,
                "canonical_pass_rate": (canonical_passed / total) * 100,
                "total": total,
            })

        if not config_stats:
            for ax in axes:
                ax.text(0.5, 0.5, "No detection data available",
                       ha="center", va="center", transform=ax.transAxes)
            plt.savefig(output_path / "10_true_vs_false_positive.png", dpi=300)
            plt.close()
            return

        stats_df = pd.DataFrame(config_stats)
        stats_df = stats_df.sort_values("config")

        # Plot 1: True Positive Rate (stacked with False Positive)
        ax1 = axes[0]
        x = range(len(stats_df))
        width = 0.35

        bars1 = ax1.bar(
            [i - width/2 for i in x],
            stats_df["true_positive_rate"],
            width,
            label="True Positive (AssertionError)",
            color="#2ecc71",
            alpha=0.85,
            edgecolor="black",
        )
        bars2 = ax1.bar(
            [i + width/2 for i in x],
            stats_df["false_positive_rate"],
            width,
            label="False Positive (Runtime Error)",
            color="#e74c3c",
            alpha=0.85,
            edgecolor="black",
        )

        ax1.set_xlabel("Configuration Type", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Detection Rate (%)", fontsize=14, fontweight="bold")
        ax1.set_title("True vs False Positive Rates", fontsize=16, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(stats_df["config"], rotation=45, ha="right")
        ax1.legend(fontsize=11)
        ax1.set_ylim(0, 100)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, height,
                        f"{height:.1f}%", ha="center", va="bottom", fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, height,
                        f"{height:.1f}%", ha="center", va="bottom", fontsize=10)

        # Plot 2: Detection breakdown pie chart (overall)
        ax2 = axes[1]
        
        total_samples = self.df.shape[0]
        true_pos_total = (self.df["is_true_positive"] == True).sum()
        false_pos_total = (self.df["is_false_positive"] == True).sum()
        false_neg_total = total_samples - true_pos_total - false_pos_total
        
        # Filter out inconclusive cases
        canonical_failed = (self.df["canonical_solution_passed"] == False).sum()
        false_neg_total -= canonical_failed

        sizes = [true_pos_total, false_pos_total, max(0, false_neg_total)]
        labels = [
            f"True Positive\n({true_pos_total})",
            f"False Positive\n({false_pos_total})",
            f"False Negative\n({max(0, false_neg_total)})",
        ]
        colors = ["#2ecc71", "#e74c3c", "#f39c12"]
        explode = (0.05, 0.05, 0.05)

        if sum(sizes) > 0:
            wedges, texts, autotexts = ax2.pie(
                sizes,
                labels=labels,
                colors=colors,
                explode=explode,
                autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
                startangle=90,
                textprops={"fontsize": 11},
            )
            ax2.set_title("Overall Bug Detection Breakdown", fontsize=16, fontweight="bold")
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center")

        plt.tight_layout()
        plt.savefig(
            output_path / "10_true_vs_false_positive.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_failure_type_distribution(self, output_path: Path) -> None:
        """Plot distribution of failure types when buggy solution fails."""
        if "buggy_failure_type" not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Count failure types
        failure_counts = self.df["buggy_failure_type"].value_counts()
        failure_counts = failure_counts[failure_counts.index != "none"]  # Exclude passed tests

        if failure_counts.empty:
            ax.text(0.5, 0.5, "No failure type data available",
                   ha="center", va="center", transform=ax.transAxes, fontsize=14)
        else:
            # Color map for failure types
            colors = {
                "assertion": "#2ecc71",  # Green - desired outcome
                "type": "#e74c3c",
                "name": "#e67e22",
                "index": "#9b59b6",
                "key": "#3498db",
                "attribute": "#1abc9c",
                "value": "#f39c12",
                "syntax": "#95a5a6",
                "import": "#7f8c8d",
                "recursion": "#c0392b",
                "zerodivision": "#d35400",
                "timeout": "#2c3e50",
                "unknown": "#bdc3c7",
            }

            bar_colors = [colors.get(ft, "#95a5a6") for ft in failure_counts.index]

            bars = ax.bar(
                range(len(failure_counts)),
                failure_counts.values,
                color=bar_colors,
                edgecolor="black",
                alpha=0.85,
            )

            ax.set_xticks(range(len(failure_counts)))
            ax.set_xticklabels(
                [ft.replace("_", " ").title() for ft in failure_counts.index],
                rotation=45,
                ha="right",
                fontsize=12,
            )

            # Highlight assertion errors
            for i, (ft, bar) in enumerate(zip(failure_counts.index, bars)):
                height = bar.get_height()
                label = f"{height}"
                if ft == "assertion":
                    label += " ✓"
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       label, ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_xlabel("Failure Type", fontsize=14, fontweight="bold")
        ax.set_ylabel("Count", fontsize=14, fontweight="bold")
        ax.set_title("Distribution of Failure Types (Buggy Solution)", 
                    fontsize=16, fontweight="bold", pad=20)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add legend explaining that assertion = true bug detection
        ax.annotate(
            "✓ = True Bug Detection",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ecc71", alpha=0.3),
        )

        plt.tight_layout()
        plt.savefig(
            output_path / "11_failure_type_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_bug_detection_by_model(self, output_path: Path) -> None:
        """Plot bug detection success rate by model."""
        if "model" not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        # Get unique models
        models = [m for m in self.df["model"].unique() if m != "unknown"]

        if not models:
            ax.text(0.5, 0.5, "No model data available",
                   ha="center", va="center", transform=ax.transAxes, fontsize=14)
            plt.savefig(output_path / "12_bug_detection_by_model.png", dpi=300)
            plt.close()
            return

        # Calculate metrics for each model
        model_stats = []
        for model in models:
            model_data = self.df[self.df["model"] == model]
            total = len(model_data)
            
            if total == 0:
                continue
            
            true_pos = (model_data["is_true_positive"] == True).sum()
            false_pos = (model_data["is_false_positive"] == True).sum()
            false_neg = total - true_pos - false_pos
            
            model_stats.append({
                "model": self._format_model_name(model),
                "true_positive": true_pos,
                "false_positive": false_pos,
                "false_negative": false_neg,
                "total": total,
                "detection_rate": (true_pos / total) * 100,
            })

        if not model_stats:
            ax.text(0.5, 0.5, "No detection data available",
                   ha="center", va="center", transform=ax.transAxes)
            plt.savefig(output_path / "12_bug_detection_by_model.png", dpi=300)
            plt.close()
            return

        stats_df = pd.DataFrame(model_stats)
        stats_df = stats_df.sort_values("detection_rate", ascending=True)

        # Horizontal bar chart
        y = range(len(stats_df))
        
        bars = ax.barh(
            y,
            stats_df["detection_rate"],
            color="#2ecc71",
            edgecolor="black",
            alpha=0.85,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(stats_df["model"], fontsize=12)
        ax.set_xlabel("True Bug Detection Rate (%)", fontsize=14, fontweight="bold")
        ax.set_title("Bug Detection Rate by Model", fontsize=16, fontweight="bold", pad=20)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        # Add value labels
        for bar, rate, total in zip(bars, stats_df["detection_rate"], stats_df["total"]):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f"{rate:.1f}% (n={total})",
                   ha="left", va="center", fontsize=11, fontweight="bold")

        plt.tight_layout()
        plt.savefig(
            output_path / "12_bug_detection_by_model.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_detection_success_summary(self, output_path: Path) -> None:
        """Plot a comprehensive summary of bug detection success metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Overall metrics summary (top-left)
        ax1 = axes[0, 0]
        total = len(self.df)
        
        if total > 0:
            metrics = {
                "True Positive\n(Assertion)": (self.df["is_true_positive"] == True).sum(),
                "False Positive\n(Runtime)": (self.df["is_false_positive"] == True).sum(),
                "Canonical\nPassed": (self.df["canonical_solution_passed"] == True).sum(),
                "Buggy\nFailed": (self.df["buggy_solution_failed"] == True).sum(),
            }
            
            x = range(len(metrics))
            colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
            bars = ax1.bar(x, list(metrics.values()), color=colors, edgecolor="black", alpha=0.85)
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(list(metrics.keys()), fontsize=11)
            
            for bar, val in zip(bars, metrics.values()):
                pct = (val / total) * 100
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
            
            ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
            ax1.set_title(f"Overall Detection Metrics (n={total})", fontsize=14, fontweight="bold")
            ax1.grid(axis="y", alpha=0.3)

        # 2. Detection rate by bug type (top-right)
        ax2 = axes[0, 1]
        
        if "bug_type" in self.df.columns:
            bug_type_rates = self.df.groupby("bug_type").agg({
                "is_true_positive": lambda x: (x == True).mean() * 100
            }).reset_index()
            bug_type_rates.columns = ["bug_type", "detection_rate"]
            bug_type_rates = bug_type_rates.sort_values("detection_rate", ascending=True)
            
            if not bug_type_rates.empty:
                y = range(len(bug_type_rates))
                bars = ax2.barh(y, bug_type_rates["detection_rate"], 
                               color="#2ecc71", edgecolor="black", alpha=0.85)
                
                ax2.set_yticks(y)
                ax2.set_yticklabels([
                    self.BUG_TYPE_DISPLAY.get(bt, bt.replace("_", " ").title())
                    for bt in bug_type_rates["bug_type"]
                ], fontsize=11)
                
                for bar, rate in zip(bars, bug_type_rates["detection_rate"]):
                    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                            f"{rate:.1f}%", ha="left", va="center", fontsize=10)
                
                ax2.set_xlabel("True Bug Detection Rate (%)", fontsize=12, fontweight="bold")
                ax2.set_title("Detection Rate by Bug Type", fontsize=14, fontweight="bold")
                ax2.set_xlim(0, 100)
                ax2.grid(axis="x", alpha=0.3)

        # 3. Coverage distribution for detected vs missed bugs (bottom-left)
        ax3 = axes[1, 0]
        
        if "code_coverage_c0_percent" in self.df.columns:
            detected = self.df[self.df["is_true_positive"] == True]["code_coverage_c0_percent"]
            missed = self.df[self.df["is_true_positive"] != True]["code_coverage_c0_percent"]
            
            data_to_plot = []
            labels = []
            if len(detected) > 0:
                data_to_plot.append(detected)
                labels.append(f"Detected\n(n={len(detected)})")
            if len(missed) > 0:
                data_to_plot.append(missed)
                labels.append(f"Missed\n(n={len(missed)})")
            
            if data_to_plot:
                bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
                colors = ["#2ecc71", "#e74c3c"]
                for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax3.set_ylabel("Code Coverage (%)", fontsize=12, fontweight="bold")
                ax3.set_title("Coverage: Detected vs Missed Bugs", fontsize=14, fontweight="bold")
                ax3.grid(axis="y", alpha=0.3)

        # 4. Success metrics text summary (bottom-right)
        ax4 = axes[1, 1]
        ax4.axis("off")
        
        if total > 0:
            true_pos = (self.df["is_true_positive"] == True).sum()
            false_pos = (self.df["is_false_positive"] == True).sum()
            
            # Calculate precision and recall-like metrics
            all_detected = true_pos + false_pos
            precision = (true_pos / all_detected * 100) if all_detected > 0 else 0
            recall = (true_pos / total) * 100
            
            summary_text = f"""
Bug Detection Summary
═══════════════════════════════════════

Total Test Cases:     {total}

True Positives:       {true_pos} ({true_pos/total*100:.1f}%)
  (Assertion failures - correct bug detection)

False Positives:      {false_pos} ({false_pos/total*100:.1f}%)
  (Runtime errors - incorrect detection)

Detection Precision:  {precision:.1f}%
  (True Positives / All Detected)

Detection Recall:     {recall:.1f}%
  (True Positives / Total Cases)

═══════════════════════════════════════

Note: True Positive = Test correctly identified 
      the bug via AssertionError
"""
            ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
                    fontsize=12, verticalalignment="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.8))

        plt.tight_layout()
        plt.savefig(
            output_path / "13_detection_success_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

