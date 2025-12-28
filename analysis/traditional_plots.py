#!/usr/bin/env python3
"""
Traditional Plotting Module

Contains the original 5 visualization methods for analyzing test generation
results across different configuration types.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class TraditionalPlots:
    """Handles the original 5 visualization methods for test result analysis."""

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

    def create_all_plots(self, output_path: Path) -> None:
        """Create all traditional visualization plots."""
        print("Creating traditional visualizations...")

        self._plot_success_rate(output_path)
        print("  ✓ Created success rate analysis")

        self._plot_code_coverage(output_path)
        print("  ✓ Created code coverage analysis")

        self._plot_cost_analysis(output_path)
        print("  ✓ Created cost analysis")

        self._plot_fix_attempts(output_path)
        print("  ✓ Created fix attempts analysis")

        self._plot_input_tokens(output_path)
        print("  ✓ Created input token analysis")

    def _plot_success_rate(self, output_path: Path) -> None:
        """Plot success rate by configuration type."""
        fig, ax = plt.subplots(figsize=(10, 6))

        success_rate = (
            self.df.groupby("config_type_display", observed=False)["success"]
            .agg(["mean", "count"])
            .reset_index()
        )
        success_rate["success_rate"] = success_rate["mean"] * 100
        success_rate = success_rate.sort_values("config_type_display")

        bars = ax.bar(success_rate["config_type_display"], success_rate["success_rate"])
        ax.set_title(
            "Success Rate by Configuration Type", fontsize=20, fontweight="bold"
        )
        ax.set_xlabel("Configuration Type", fontsize=16, fontweight="bold")
        ax.set_ylabel("Success Rate (%)", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 100)

        # Add value labels on bars
        for bar, rate, count in zip(
            bars, success_rate["success_rate"], success_rate["count"]
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / "1_success_rate.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_code_coverage(self, output_path: Path) -> None:
        """Plot code coverage by configuration type."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar plot of mean coverage
        coverage_stats = (
            self.df.groupby("config_type_display", observed=False)[
                "code_coverage_percent"
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        coverage_stats = coverage_stats.sort_values("config_type_display")
        bars = ax.bar(
            coverage_stats["config_type_display"],
            coverage_stats["mean"],
            yerr=coverage_stats["std"],
            capsize=5,
        )
        ax.set_title(
            "Average Code Coverage by Configuration", fontsize=20, fontweight="bold"
        )
        ax.set_xlabel("Configuration Type", fontsize=16, fontweight="bold")
        ax.set_ylabel("Average Code Coverage (%)", fontsize=16, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        # Code coverage ranges from 0-100%
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, mean, count in zip(
            bars, coverage_stats["mean"], coverage_stats["count"]
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{mean:.1f}%",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path / "2_code_coverage.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_cost_analysis(self, output_path: Path) -> None:
        """Plot cost analysis by configuration type."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Average cost bar plot
        cost_stats = (
            self.df.groupby("config_type_display", observed=False)["total_cost_usd"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        cost_stats = cost_stats.sort_values("config_type_display")
        bars = ax.bar(
            cost_stats["config_type_display"],
            cost_stats["mean"],
            yerr=cost_stats["std"],
            capsize=5,
        )
        ax.set_title(
            "Average Total Cost by Configuration", fontsize=20, fontweight="bold"
        )
        ax.set_xlabel("Configuration Type", fontsize=16, fontweight="bold")
        ax.set_ylabel("Average Total Cost (USD)", fontsize=16, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        # Cost cannot be negative - set y-axis to start from 0
        ax.set_ylim(bottom=0)

        # Add value labels
        for bar, mean, count in zip(bars, cost_stats["mean"], cost_stats["count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"${mean:.4f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path / "3_cost_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_fix_attempts(self, output_path: Path) -> None:
        """Plot fix attempts needed by configuration type."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Average fix attempts
        fix_stats = (
            self.df.groupby("config_type_display", observed=False)["fix_attempts_used"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        fix_stats = fix_stats.sort_values("config_type_display")
        bars = ax.bar(
            fix_stats["config_type_display"],
            fix_stats["mean"],
            yerr=fix_stats["std"],
            capsize=5,
        )
        ax.set_title(
            "Average Fix Attempts by Configuration", fontsize=20, fontweight="bold"
        )
        ax.set_xlabel("Configuration Type", fontsize=16, fontweight="bold")
        ax.set_ylabel("Average Fix Attempts", fontsize=16, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        # Fix attempts cannot be negative - set y-axis to start from 0
        ax.set_ylim(bottom=0)

        # Add value labels
        for bar, mean, count in zip(bars, fix_stats["mean"], fix_stats["count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path / "4_fix_attempts.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_input_tokens(self, output_path: Path) -> None:
        """Plot input token usage by configuration type."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar plot for average
        token_stats = (
            self.df.groupby("config_type_display", observed=False)["total_input_tokens"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        token_stats = token_stats.sort_values("config_type_display")
        bars = ax.bar(
            token_stats["config_type_display"],
            token_stats["mean"],
            yerr=token_stats["std"],
            capsize=5,
        )
        ax.set_title(
            "Average Input Tokens by Configuration", fontsize=20, fontweight="bold"
        )
        ax.set_xlabel("Configuration Type", fontsize=16, fontweight="bold")
        ax.set_ylabel("Average Input Tokens", fontsize=16, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        # Set y-axis to start at 0 for consistency
        ax.set_ylim(bottom=0)

        # Add value labels
        for bar, mean, count in zip(bars, token_stats["mean"], token_stats["count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{mean:.0f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path / "5_input_tokens.png", dpi=300, bbox_inches="tight")
        plt.close()
